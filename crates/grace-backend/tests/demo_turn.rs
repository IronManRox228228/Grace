//! End-to-end proof that `GRACE_BACKEND=rust`'s wiring (as
//! `src-tauri/src/backend.rs` sets it up) actually runs a real turn: a real
//! client connects over a real WebSocket, sends the same `{"type": "wake"}`
//! message the renderer sends, and the real `grace_core` turn engine's
//! event sequence comes back over that same socket. No fakes at the
//! transport layer - only the turn's own external edges (LLM, STT) are
//! stand-ins, exactly as `run_demo_turn`'s doc comment describes.

use futures_util::{SinkExt, StreamExt};
use grace_backend::{run_demo_turn, WsEventServer};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

#[tokio::test]
async fn a_wake_message_over_the_real_socket_runs_a_real_turn() {
    let server = WsEventServer::new("127.0.0.1", 0, "");
    let addr = server.serve().await.unwrap();

    let wake_server = Arc::clone(&server);
    server
        .set_on_wake(Arc::new(move || {
            let server = Arc::clone(&wake_server);
            std::thread::spawn(move || {
                run_demo_turn(&server);
            });
        }))
        .await;

    let url = format!("ws://{addr}/");
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    // Let the server register this client's subscription before waking it.
    tokio::time::sleep(Duration::from_millis(50)).await;

    ws.send(Message::Text(r#"{"type": "wake"}"#.to_string().into())).await.unwrap();

    let mut event_types = Vec::new();
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for the turn's events")
            .expect("stream ended")
            .expect("websocket error");
        let Message::Text(text) = msg else { continue };
        let value: Value = serde_json::from_str(&text).unwrap();
        let event_type = value["type"].as_str().unwrap().to_string();
        let done = event_type == "ConversationFinished";
        event_types.push(event_type);
        if done {
            break;
        }
    }

    assert_eq!(
        event_types,
        vec![
            "WakeWordDetected",
            "ListeningStarted",
            "FinalTranscript",
            "ListeningStopped",
            "UnderstandingStarted",
            "UnderstandingFinished",
            "ToolExecutionStarted",
            "ToolExecutionFinished",
            "SpeechStarted",
            "ResponseChunk",
            "SpeechChunk",
            "SpeechFinished",
            "ConversationFinished",
        ]
    );
}
