//! A minimal seam for emitting `grace_contract::GraceEvent`s from pure
//! orchestration code (the response generator, and eventually the turn
//! engine), so tests can assert on the exact emitted sequence without a
//! real WebSocket. The production wiring (`src-tauri`/`grace-backend`)
//! implements this trait over `grace_backend::WsEventServer::emit`.

use grace_contract::GraceEvent;

pub trait EventSink: Send {
    fn emit(&mut self, event: GraceEvent);
}

/// Records every emitted event in order, for tests to assert against.
#[derive(Default)]
pub struct RecordingEventSink {
    pub events: Vec<GraceEvent>,
}

impl EventSink for RecordingEventSink {
    fn emit(&mut self, event: GraceEvent) {
        self.events.push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_sink_keeps_emitted_events_in_order() {
        let mut sink = RecordingEventSink::default();
        sink.emit(GraceEvent::Idle);
        sink.emit(GraceEvent::WakeWordDetected);
        assert_eq!(sink.events, vec![GraceEvent::Idle, GraceEvent::WakeWordDetected]);
    }
}
