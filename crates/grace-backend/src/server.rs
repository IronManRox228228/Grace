//! The Rust event server: binds `127.0.0.1:<port>`, enforces the Origin
//! allowlist from `origin.rs` on every handshake, and broadcasts
//! `grace_contract::GraceEvent` JSON to every connected client - the same
//! contract `src/grace/ws_server.py`'s `WsEventServer.emit` serves.
//!
//! **Scope of this phase**: this server does not yet drive a turn. It is
//! the transport ported ahead of the agent loop / dispatcher / perception
//! pipeline, so the Tauri shell has something real to point at and the
//! Origin-allowlist ship-blocker fix is enforced identically regardless of
//! which backend answers the socket. See PORT_STATUS.md.

use crate::origin::{default_allowed_origins, is_origin_allowed, parse_extra_origins};
use futures_util::{SinkExt, StreamExt};
use grace_contract::GraceEvent;
use std::collections::BTreeSet;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, Mutex};
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::http;
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::Message;

/// A callback invoked whenever a connected client sends `{"type": "wake"}` -
/// mirrors `WsEventServer.set_on_wake`.
pub type WakeCallback = Arc<dyn Fn() + Send + Sync>;

/// Mirrors `WsEventServer`: owns the allowed-origins set and broadcasts
/// `GraceEvent`s to every connected client.
pub struct WsEventServer {
    host: String,
    port: u16,
    allowed_origins: BTreeSet<String>,
    events_tx: broadcast::Sender<GraceEvent>,
    on_wake: Mutex<Option<WakeCallback>>,
    client_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl WsEventServer {
    pub fn new(host: impl Into<String>, port: u16, allowed_origins_csv: &str) -> Arc<Self> {
        let extra = parse_extra_origins(allowed_origins_csv);
        let allowed_origins: BTreeSet<String> = default_allowed_origins().union(&extra).cloned().collect();
        let (events_tx, _rx) = broadcast::channel(256);
        Arc::new(Self {
            host: host.into(),
            port,
            allowed_origins,
            events_tx,
            on_wake: Mutex::new(None),
            client_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    pub fn allowed_origins(&self) -> &BTreeSet<String> {
        &self.allowed_origins
    }

    pub async fn set_on_wake(&self, callback: WakeCallback) {
        *self.on_wake.lock().await = Some(callback);
    }

    pub fn is_connected(&self) -> bool {
        self.client_count.load(std::sync::atomic::Ordering::SeqCst) > 0
    }

    /// Broadcasts one event to every connected client. Mirrors
    /// `WsEventServer.emit` - silently a no-op with no clients connected,
    /// same as the Python server dropping to an empty client set.
    pub fn emit(&self, event: GraceEvent) {
        let _ = self.events_tx.send(event);
    }

    /// Binds the listener and serves forever (until the returned future is
    /// dropped/cancelled). Returns the bound address, so callers that bind
    /// to port 0 in tests can discover the actual port.
    pub async fn serve(self: &Arc<Self>) -> anyhow::Result<std::net::SocketAddr> {
        let listener = TcpListener::bind((self.host.as_str(), self.port)).await?;
        let addr = listener.local_addr()?;
        let this = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                let (stream, _peer) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                let this = Arc::clone(&this);
                tokio::spawn(async move {
                    let _ = this.handle_connection(stream).await;
                });
            }
        });
        Ok(addr)
    }

    async fn handle_connection(self: Arc<Self>, stream: TcpStream) -> anyhow::Result<()> {
        let allowed = self.allowed_origins.clone();
        let callback = move |req: &Request, response: Response| {
            let origin = req
                .headers()
                .get("Origin")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            if is_origin_allowed(origin.as_deref(), &allowed) {
                Ok(response)
            } else {
                let resp: ErrorResponse = http::Response::builder()
                    .status(StatusCode::FORBIDDEN)
                    .body(Some("Origin not allowed".to_string()))
                    .expect("a fixed status + body always builds a valid Response");
                Err(resp)
            }
        };

        let ws_stream = match tokio_tungstenite::accept_hdr_async(stream, callback).await {
            Ok(s) => s,
            Err(_) => return Ok(()), // rejected handshake or client disconnect; nothing to serve
        };

        self.client_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (mut write, mut read) = ws_stream.split();
        let mut events_rx = self.events_tx.subscribe();

        loop {
            tokio::select! {
                event = events_rx.recv() => {
                    match event {
                        Ok(event) => {
                            let Ok(json) = serde_json::to_string(&event) else { continue };
                            if write.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
                incoming = read.next() => {
                    match incoming {
                        Some(Ok(Message::Text(text))) => {
                            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                                if value.get("type").and_then(serde_json::Value::as_str) == Some("wake") {
                                    if let Some(cb) = self.on_wake.lock().await.as_ref() {
                                        cb();
                                    }
                                }
                            }
                        }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(_)) => break,
                        _ => {}
                    }
                }
            }
        }

        self.client_count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    async fn connect_with_origin(
        addr: std::net::SocketAddr,
        origin: Option<&str>,
    ) -> Result<(), tokio_tungstenite::tungstenite::Error> {
        let url = format!("ws://{addr}/");
        let mut request = url.into_client_request().unwrap();
        if let Some(origin) = origin {
            request
                .headers_mut()
                .insert("Origin", origin.parse().unwrap());
        }
        tokio_tungstenite::connect_async(request).await.map(|_| ())
    }

    #[tokio::test]
    async fn handshake_from_a_disallowed_origin_is_rejected() {
        let server = WsEventServer::new("127.0.0.1", 0, "");
        let addr = server.serve().await.unwrap();

        let result = connect_with_origin(addr, Some("http://evil.example")).await;
        assert!(result.is_err(), "expected the handshake to be rejected");
    }

    #[tokio::test]
    async fn handshake_from_an_allowed_origin_succeeds() {
        let server = WsEventServer::new("127.0.0.1", 0, "");
        let addr = server.serve().await.unwrap();

        let result = connect_with_origin(addr, Some("http://localhost:5173")).await;
        assert!(result.is_ok(), "expected the handshake to succeed: {result:?}");
    }

    #[tokio::test]
    async fn handshake_with_no_origin_header_succeeds() {
        let server = WsEventServer::new("127.0.0.1", 0, "");
        let addr = server.serve().await.unwrap();

        let result = connect_with_origin(addr, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn ws_allowed_origins_override_extends_the_allowlist() {
        let server = WsEventServer::new("127.0.0.1", 0, "http://example.test");
        let addr = server.serve().await.unwrap();

        assert!(connect_with_origin(addr, Some("http://example.test")).await.is_ok());
    }

    #[tokio::test]
    async fn an_emitted_event_reaches_a_connected_client() {
        let server = WsEventServer::new("127.0.0.1", 0, "");
        let addr = server.serve().await.unwrap();

        let url = format!("ws://{addr}/");
        let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        // Give the server a moment to register the subscription before
        // emitting, since the accept loop runs on a spawned task.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        server.emit(GraceEvent::Idle);

        let msg = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .expect("timed out waiting for the event")
            .expect("stream ended")
            .expect("websocket error");
        let Message::Text(text) = msg else {
            panic!("expected a text frame, got {msg:?}");
        };
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["type"], "Idle");
    }
}
