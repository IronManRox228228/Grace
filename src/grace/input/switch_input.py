"""Switch accessibility input handler (PLAN.md §7, Task 10).

Provides single-switch accessibility control for motor dysfunction users:
- Single press = YES (confirm action)
- Double press (within configurable window) = NO (reject action)
- Long press (held >= threshold) = STOP (trigger voice stop reflex)

Operates in every confirmation dialog and Set-of-Marks ("4 or 7?") fallback.
Strictly coexists with voice: whichever modality answers first wins; the other
modality is cleanly ignored for that turn.
"""

from __future__ import annotations

from enum import Enum
import logging
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger("grace.input.switch")


class SwitchAction(str, Enum):
    YES = "yes"
    NO = "no"
    STOP = "stop"


class SwitchInputHandler:
    """Processes switch press timings into discrete accessibility actions."""

    def __init__(
        self,
        double_press_window_ms: float = 400.0,
        long_press_threshold_ms: float = 800.0,
        switch_key: str = "space",
        on_action: Optional[Callable[[SwitchAction], None]] = None,
    ) -> None:
        self.double_press_window_ms = float(double_press_window_ms)
        self.long_press_threshold_ms = float(long_press_threshold_ms)
        self.switch_key = str(switch_key).lower()
        self.on_action = on_action

        self._lock = threading.Lock()
        self._press_start_time: Optional[float] = None
        self._pending_timer: Optional[threading.Timer] = None
        self._long_press_timer: Optional[threading.Timer] = None
        self._long_press_fired = False
        self._first_up_time: Optional[float] = None
        self._is_active = True

    def reset(self) -> None:
        """Reset internal timers and pending state."""
        with self._lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
            if self._long_press_timer is not None:
                self._long_press_timer.cancel()
                self._long_press_timer = None
            self._press_start_time = None
            self._long_press_fired = False
            self._first_up_time = None

    def trigger_action(self, action: SwitchAction) -> None:
        """Directly trigger an action (used for simulation and programmatic dispatch)."""
        self.reset()
        self._emit_action(action)

    def handle_key(
        self, key: str, is_down: bool, timestamp: Optional[float] = None
    ) -> Optional[SwitchAction]:
        """Process keyboard/switch key state changes against configured switch_key."""
        if not key or key.strip().lower() != self.switch_key:
            return None
        if is_down:
            self.press_down(timestamp)
            return None
        return self.press_up(timestamp)

    def press_down(self, timestamp: Optional[float] = None) -> None:
        """Handle switch button press down event."""
        t = timestamp if timestamp is not None else time.perf_counter()
        with self._lock:
            self._press_start_time = t
            self._long_press_fired = False
            if self._long_press_timer is not None:
                self._long_press_timer.cancel()

            # Schedule long press STOP if held >= threshold
            def _fire_long():
                with self._lock:
                    if self._press_start_time is not None and not self._long_press_fired:
                        self._long_press_fired = True
                        if self._pending_timer is not None:
                            self._pending_timer.cancel()
                            self._pending_timer = None
                        self._first_up_time = None
                        self._emit_action(SwitchAction.STOP)

            delay_sec = self.long_press_threshold_ms / 1000.0
            self._long_press_timer = threading.Timer(delay_sec, _fire_long)
            self._long_press_timer.daemon = True
            self._long_press_timer.start()

    def press_up(self, timestamp: Optional[float] = None) -> Optional[SwitchAction]:
        """Handle switch button release event. Returns action if synchronously resolved."""
        t = timestamp if timestamp is not None else time.perf_counter()
        with self._lock:
            if self._long_press_timer is not None:
                self._long_press_timer.cancel()
                self._long_press_timer = None

            if self._long_press_fired:
                self._press_start_time = None
                return SwitchAction.STOP

            if self._press_start_time is None:
                return None

            duration_ms = (t - self._press_start_time) * 1000.0
            self._press_start_time = None

            # 1. Long Press check -> STOP action immediately
            if duration_ms >= self.long_press_threshold_ms:
                if self._pending_timer is not None:
                    self._pending_timer.cancel()
                    self._pending_timer = None
                self._first_up_time = None
                self._emit_action(SwitchAction.STOP)
                return SwitchAction.STOP

            # 2. Double Press check
            if self._first_up_time is not None:
                interval_ms = (t - self._first_up_time) * 1000.0
                if interval_ms <= self.double_press_window_ms:
                    # Second press within window -> NO action
                    if self._pending_timer is not None:
                        self._pending_timer.cancel()
                        self._pending_timer = None
                    self._first_up_time = None
                    self._emit_action(SwitchAction.NO)
                    return SwitchAction.NO

            # 3. First Short Press -> start timer for YES action
            self._first_up_time = t
            if self._pending_timer is not None:
                self._pending_timer.cancel()

            # Schedule delayed YES in case no second press occurs
            def _fire_single_yes():
                with self._lock:
                    self._pending_timer = None
                    self._first_up_time = None
                    self._emit_action(SwitchAction.YES)

            delay_sec = self.double_press_window_ms / 1000.0
            self._pending_timer = threading.Timer(delay_sec, _fire_single_yes)
            self._pending_timer.daemon = True
            self._pending_timer.start()
            return None

    def _emit_action(self, action: SwitchAction) -> None:
        logger.info("SwitchInputHandler triggered action: %s", action.value)
        if self.on_action:
            try:
                self.on_action(action)
            except Exception as e:
                logger.error("Error in switch action callback: %s", e)


class VoiceSwitchRaceResolver:
    """Resolves race between Voice and Switch input. Whichever answers first wins."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resolved = False
        self._winner: Optional[str] = None
        self._result: Optional[Any] = None

    @property
    def is_resolved(self) -> bool:
        with self._lock:
            return self._resolved

    @property
    def winner(self) -> Optional[str]:
        with self._lock:
            return self._winner

    @property
    def result(self) -> Optional[Any]:
        with self._lock:
            return self._result

    def resolve_voice(self, answer: str) -> bool:
        """Attempt to resolve the pending question via voice. Returns True if won."""
        with self._lock:
            if self._resolved:
                return False
            self._resolved = True
            self._winner = "voice"
            self._result = answer
            logger.info("Voice won the confirmation race: '%s'", answer)
            return True

    def resolve_switch(self, action: SwitchAction) -> bool:
        """Attempt to resolve the pending question via switch. Returns True if won."""
        with self._lock:
            if self._resolved:
                return False
            self._resolved = True
            self._winner = "switch"
            # Map switch action to confirmation answer
            if action == SwitchAction.YES:
                self._result = "yes"
            elif action == SwitchAction.NO:
                self._result = "no"
            elif action == SwitchAction.STOP:
                self._result = "stop"
            logger.info("Switch won the confirmation race: '%s'", action.value)
            return True


class MarkScanner:
    """Cyclic scanner for Set-of-Marks ambiguity ('4 or 7?').

    Periodically steps through visual mark choices, allowing switch selection.
    """

    def __init__(
        self,
        options: List[str],
        scan_rate_ms: float = 1500.0,
        on_highlight: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        self.options = list(options)
        self.scan_rate_ms = float(scan_rate_ms)
        self.on_highlight = on_highlight

        self._lock = threading.Lock()
        self._current_index = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def current_index(self) -> int:
        with self._lock:
            return self._current_index

    @property
    def current_option(self) -> Optional[str]:
        with self._lock:
            if not self.options:
                return None
            return self.options[self._current_index]

    def start(self) -> None:
        """Start cyclic mark scanner loop."""
        if not self.options or self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scan_loop, name="MarkScannerThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop cyclic scanner loop."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None

    def select_current(self) -> Optional[str]:
        """User pressed switch: returns currently highlighted mark and stops scan."""
        with self._lock:
            if not self.options:
                return None
            selected = self.options[self._current_index]
        self.stop()
        logger.info("MarkScanner selected mark: '%s'", selected)
        return selected

    def _scan_loop(self) -> None:
        interval_sec = self.scan_rate_ms / 1000.0
        while not self._stop_event.is_set():
            with self._lock:
                idx = self._current_index
                val = self.options[idx]
            if self.on_highlight:
                try:
                    self.on_highlight(idx, val)
                except Exception as e:
                    logger.debug("Error in mark highlight callback: %s", e)

            if self._stop_event.wait(timeout=interval_sec):
                break

            with self._lock:
                self._current_index = (self._current_index + 1) % len(self.options)
