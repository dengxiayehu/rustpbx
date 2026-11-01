//! Test harness for [`CallApp`] implementations.
//!
//! Provides [`MockCallStack`]: a fully assembled in-memory call stack that
//! runs a `CallApp` without any real SIP session, media, or database.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use crate::call::app::testing::MockCallStack;
//! use crate::call::domain::CallCommand;
//!
//! let mut stack = MockCallStack::run(Box::new(MyApp::new()), "1001", "1002");
//!
//! // App answers on enter → assert Accept
//! stack.assert_cmd(50, "answer", |c| matches!(c, CallCommand::Accept)).await;
//!
//! // Simulate audio playback finishing
//! stack.audio_complete("default");
//!
//! // App should now hang up
//! stack.assert_cmd(50, "hangup", |c| matches!(c, CallCommand::Hangup { .. })).await;
//! ```

use super::{AppEventLoop, CallApp, CallController, ControllerEvent, RecordingInfo};
use crate::call::app::{ApplicationContext, CallInfo};
use crate::call::domain::CallCommand;
use crate::config::Config;
use crate::proxy::proxy_call::sip_session::SipSessionHandle;
use chrono::Utc;
use sea_orm::DatabaseConnection;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::instrument::WithSubscriber;

/// Type alias for mock call stack command receiver.
pub type MockCmdRx = tokio::sync::mpsc::Receiver<CallCommand>;

/// Minimal [`ApplicationContext`] for tests: session `"test-session"`, inbound
/// call from `caller` to `callee`, no DB, default config.
pub fn test_context_for(caller: &str, callee: &str) -> ApplicationContext {
    test_context_with_config(caller, callee, Arc::new(Config::default()))
}

/// [`test_context_for`] with the conventional fixture identity
/// (caller `"1001"`, callee `"2000"`).
pub fn test_context() -> ApplicationContext {
    test_context_for("1001", "2000")
}

/// [`test_context_for`] with a specific [`Config`] (e.g. proxy settings
/// parsed from TOML in queue tests).
pub fn test_context_with_config(
    caller: &str,
    callee: &str,
    config: Arc<Config>,
) -> ApplicationContext {
    ApplicationContext::new(
        DatabaseConnection::default(),
        CallInfo {
            session_id: "test-session".into(),
            caller: caller.into(),
            callee: callee.into(),
            direction: "inbound".into(),
            started_at: Utc::now(),
            sip_headers: HashMap::new(),
            route_name: None,
        },
        config,
        reqwest::Client::new(),
    )
}

/// In-memory [`std::io::Write`] sink that captures tracing output for log
/// assertions. Clone it freely — all clones share the same buffer.
#[derive(Clone)]
pub struct MemoryLogWriter(pub Arc<std::sync::Mutex<Vec<u8>>>);

impl MemoryLogWriter {
    pub fn new() -> Self {
        Self(Arc::new(std::sync::Mutex::new(Vec::new())))
    }

    /// Full captured log text.
    pub fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

impl Default for MemoryLogWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::io::Write for MemoryLogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Build a `tracing` fmt subscriber writing into `writer` (no timestamps,
/// no ANSI). Wrap the returned subscriber in `set_default` / `with_default`
/// as the caller requires.
pub fn log_capture_subscriber(
    writer: MemoryLogWriter,
) -> impl tracing::Subscriber + Send + Sync + 'static {
    let w = writer.clone();
    tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(move || w.clone())
        .finish()
}

/// Spawn a one-shot axum HTTP server that answers every method on `path`
/// with the static JSON `body`. Returns the full URL. Test scaffolding
/// shared by the IVR webhook/provider suites.
pub async fn spawn_static_json_server(path: &str, body: serde_json::Value) -> String {
    use axum::{Json, Router, routing::any};

    let router = Router::new().route(
        path,
        any(move || {
            let body = body.clone();
            async move { Json(body) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().unwrap();
    crate::utils::spawn(async move {
        axum::serve(listener, router).await.ok();
    });
    format!("http://{}:{}{}", addr.ip(), addr.port(), path)
}

/// Fully assembled, in-memory [`CallApp`] harness — no SIP stack required.
///
/// The harness wires up:
/// - A real [`SipSessionHandle`] backed by an in-memory channel (`cmd_rx`).
/// - A synthetic event channel (`event_tx`) for injecting DTMF, audio events, etc.
/// - A real [`AppEventLoop`] running the app in a background Tokio task.
pub struct MockCallStack {
    /// Inject SIP-originated events into the running app.
    event_tx: mpsc::UnboundedSender<ControllerEvent>,
    /// Observe [`CallCommand`]s emitted by the app toward the SIP layer.
    cmd_rx: MockCmdRx,
    /// Queue-meta updates (`CallCommand::UpdateQueueMeta`) recorded here —
    /// `next_cmd` skips them (fire-and-forget metadata, like `Trace`) so
    /// existing command-sequence assertions keep working while overflow tests
    /// can still assert on the recorded sequence.
    pub queue_meta_updates: std::sync::Mutex<Vec<(Option<String>, Option<String>, Option<String>)>>,
    /// Recorded [`CallCommand::PinAgentMeta`] payloads (see [`Self::next_cmd`]).
    pub pinned_agent_meta: std::sync::Mutex<Vec<(Option<String>, Option<String>)>>,
    /// Cancel token wired to the AppEventLoop's child token.
    cancel: CancellationToken,
    /// Background task running the AppEventLoop.
    join_handle: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl MockCallStack {
    /// Build the call stack and start the app in a background task.
    ///
    /// # Arguments
    /// * `app`    – The `CallApp` under test.
    /// * `caller` – Simulated caller-ID (e.g. `"1001"` or `"sip:alice@test"`).
    /// * `callee` – Simulated callee-ID (e.g. `"1002"`).
    pub fn run(app: Box<dyn CallApp>, caller: &str, callee: &str) -> Self {
        Self::run_with_context(app, test_context_for(caller, callee))
    }

    pub fn run_with_context(app: Box<dyn CallApp>, ctx: ApplicationContext) -> Self {
        // Real SipSessionHandle — backed only by channels, no SIP socket.
        let (session_tx, mut session_rx) = mpsc::channel(256);
        let handle = SipSessionHandle::new_for_test("test-session", session_tx);
        // Request/reply probes are answered here so the app never waits on
        // the test reading its commands; everything else is forwarded.
        let (cmd_tx, cmd_rx) = mpsc::channel(256);
        crate::utils::spawn(async move {
            while let Some(cmd) = session_rx.recv().await {
                match cmd {
                    // No offline targets in the mock stack: every probe
                    // reports the target as reachable.
                    CallCommand::ProbeOfflineTarget { target: _, reply } => {
                        let _ = reply.send(false);
                    }
                    other => {
                        if cmd_tx.send(other).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });

        // Synthetic event channel: test → controller.
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<ControllerEvent>();

        // Controller + companion timer-fire channel for AppEventLoop.
        let (controller, timer_rx) = CallController::new(handle, event_rx);

        let cancel = CancellationToken::new();
        let event_loop = AppEventLoop::new(app, controller, ctx, cancel.child_token(), timer_rx);

        let join_handle = crate::utils::spawn(event_loop.run().with_current_subscriber());

        Self {
            event_tx,
            cmd_rx,
            queue_meta_updates: std::sync::Mutex::new(Vec::new()),
            pinned_agent_meta: std::sync::Mutex::new(Vec::new()),
            cancel,
            join_handle,
        }
    }

    // ── Event injection ───────────────────────────────────────────────────────

    /// Inject a DTMF digit received from the remote party.
    pub fn dtmf(&self, digit: impl Into<String>) -> &Self {
        let _ = self
            .event_tx
            .send(ControllerEvent::DtmfReceived(digit.into()));
        self
    }

    /// Inject an audio-playback-complete event.
    pub fn audio_complete(&self, track_id: impl Into<String>) -> &Self {
        let _ = self.event_tx.send(ControllerEvent::AudioComplete {
            track_id: track_id.into(),
            interrupted: false,
        });
        self
    }

    /// Inject an interrupted audio event (e.g. barge-in).
    pub fn audio_interrupted(&self, track_id: impl Into<String>) -> &Self {
        let _ = self.event_tx.send(ControllerEvent::AudioComplete {
            track_id: track_id.into(),
            interrupted: true,
        });
        self
    }

    /// Inject a recording-complete event.
    pub fn record_complete(
        &self,
        path: impl Into<String>,
        duration: Duration,
        size_bytes: u64,
    ) -> &Self {
        let _ = self
            .event_tx
            .send(ControllerEvent::RecordingComplete(RecordingInfo {
                path: path.into(),
                duration,
                size_bytes,
            }));
        self
    }

    /// Inject a remote-hangup event (no specific reason).
    pub fn remote_hangup(&self) -> &Self {
        let _ = self.event_tx.send(ControllerEvent::Hangup(None));
        self
    }

    /// Inject a custom event.
    pub fn custom(&self, name: impl Into<String>, data: serde_json::Value) -> &Self {
        let _ = self
            .event_tx
            .send(ControllerEvent::Custom(name.into(), data));
        self
    }

    /// Return a clone of the event sender so callers can inject events
    /// asynchronously (e.g. from a background task simulating playback
    /// completion).
    pub fn event_sender(&self) -> mpsc::UnboundedSender<ControllerEvent> {
        self.event_tx.clone()
    }

    // ── Command observation ───────────────────────────────────────────────────

    /// Wait up to `timeout_ms` milliseconds for the next actionable
    /// [`CallCommand`] sent by the app. Diagnostic `Trace` / `ReportCallError`
    /// events are skipped so existing command-sequence assertions keep working.
    ///
    /// Returns `None` on timeout.
    pub async fn next_cmd(&mut self, timeout_ms: u64) -> Option<CallCommand> {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            match tokio::time::timeout_at(deadline, self.cmd_rx.recv()).await {
                Ok(Some(CallCommand::Trace { .. }))
                | Ok(Some(CallCommand::ReportCallError { .. })) => continue,
                Ok(Some(cmd @ CallCommand::PinAgentMeta { .. })) => {
                    // Fire-and-forget attribution metadata (like
                    // UpdateQueueMeta): recorded, never part of command
                    // sequence assertions.
                    if let CallCommand::PinAgentMeta { agent_id, agent_name } = &cmd {
                        self.pinned_agent_meta.lock().unwrap().push((
                            agent_id.clone(),
                            agent_name.clone(),
                        ));
                    }
                    continue;
                }
                Ok(Some(cmd @ CallCommand::UpdateQueueMeta { .. })) => {
                    if let CallCommand::UpdateQueueMeta {
                        queue_name,
                        queue_label,
                        skill_group_id,
                    } = &cmd
                    {
                        self.queue_meta_updates.lock().unwrap().push((
                            queue_name.clone(),
                            queue_label.clone(),
                            skill_group_id.clone(),
                        ));
                    }
                    continue;
                }
                Ok(Some(cmd)) => return Some(cmd),
                Ok(None) | Err(_) => return None,
            }
        }
    }

    /// Assert the next command satisfies `matcher` within `timeout_ms` ms.
    ///
    /// # Panics
    /// If the timeout elapses or the matcher returns `false`.
    pub async fn assert_cmd<F>(&mut self, timeout_ms: u64, label: &str, matcher: F)
    where
        F: FnOnce(&CallCommand) -> bool,
    {
        let cmd = self
            .next_cmd(timeout_ms)
            .await
            .unwrap_or_else(|| panic!("timed out waiting for command '{label}'"));
        assert!(matcher(&cmd), "command mismatch for '{label}': got {cmd:?}");
    }

    /// Drain all immediately-available commands without blocking.
    /// `UpdateQueueMeta` commands are recorded (see [`Self::queue_meta_updates`])
    /// and skipped, mirroring [`Self::next_cmd`].
    pub fn drain_cmds(&mut self) -> Vec<CallCommand> {
        let mut out = Vec::new();
        while let Ok(cmd) = self.cmd_rx.try_recv() {
            match cmd {
                CallCommand::UpdateQueueMeta {
                    queue_name,
                    queue_label,
                    skill_group_id,
                } => {
                    self.queue_meta_updates.lock().unwrap().push((
                        queue_name,
                        queue_label,
                        skill_group_id,
                    ));
                }
                // Attribution metadata; recorded, not part of the sequence.
                CallCommand::PinAgentMeta { agent_id, agent_name } => {
                    self.pinned_agent_meta
                        .lock()
                        .unwrap()
                        .push((agent_id, agent_name));
                }
                // Diagnostic side-channel; not part of the app action sequence.
                CallCommand::ReportCallError { .. } => {}
                other => out.push(other),
            }
        }
        out
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Cancel the running event loop (simulate system shutdown).
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Wait for the event loop task to complete and surface any error.
    pub async fn join(self) -> anyhow::Result<()> {
        self.join_handle.await.expect("MockCallStack task panicked")
    }

    /// Inject a timeout event.
    pub fn timeout(&self, id: impl Into<String>) -> &Self {
        let _ = self.event_tx.send(ControllerEvent::Timeout(id.into()));
        self
    }

    /// Simulate entering the app (calls on_enter and processes initial commands).
    pub async fn enter(&mut self) {
        // Wait for initial commands from on_enter
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
