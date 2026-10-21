# Grace — progress log

## Incident fix

Session of 2026-10-22, following the 2026-10-17 desktop-input incident. Scope:
step 1 of `incident-fix-gemini.md` only (run the serial suite with `-rf`, name
and fix the one failing test, push), then this report. Steps 2–4 were **not**
started; `-n auto` was **not** run.

### Commits

- `be8537f` — `test: block os.startfile in tests` — already committed **and
  pushed** on 2026-10-17; the conftest `os.startfile` block was not left
  uncommitted as the task brief assumed.
- `311323e` — `test: make desktop-input guard fail loudly, mock every leaking
  test` — also already committed and pushed on 2026-10-17. It added
  `tests/test_desktop_guard.py`, the `PYTEST_CURRENT_TEST` attribution and the
  `pytest_sessionfinish` dump to `%TEMP%\grace-blocked-calls.txt`, and fixed
  four leaking tests (listed below).
- `2805785` — `test: mock win32gui focus calls in Win32Driver click test` —
  this session.
- this commit — `docs(progress): record incident-fix report` — this session
  (this file).

### The one failing test (serial suite, `-rf`)

`FAILED tests/test_win32_driver.py::TestWin32Driver::test_win32_driver_click_at_does_not_raise`

The test called `Win32Driver.click_at(100, 100)` with no mocks on the focus
step, so it reached real desktop calls: `win32gui.WindowFromPoint((100, 100))`
returned a **real window on the user's desktop** (hwnd 1050324), real
`win32gui.IsWindowVisible` / `win32gui.GetForegroundWindow` decided focus, and
the conftest guard's blocks of `win32gui.ShowWindow` /
`win32gui.SetForegroundWindow` (recorded in `%TEMP%\grace-blocked-calls.txt`)
were swallowed by the driver's `except Exception`, so `click_at` returned
`False`. Whether the test passed depended on which real window sat at
(100, 100) — which is why it was green on 2026-10-17 and red today.

Fix (commit `2805785`): the test now patches the exact calls it reaches —
`win32gui.WindowFromPoint` → fake hwnd `0x1234`, `win32gui.IsWindowVisible` →
`True`, `win32gui.GetForegroundWindow` → the same fake hwnd — so the focus
step is skipped and nothing real is read or driven. The conftest guard was not
weakened or touched.

Leaking tests fixed earlier, in `311323e` (verified this session via `git show`),
with the real call each reached:

- `tests/test_app_indexer.py::TestAppIndexer::test_launch_nonexistent` →
  `os.startfile` (via `AppIndexer.launch`)
- `tests/test_break.py::TestDispatcherBreak::test_open_calculator_without_cua`
  → `os.startfile` (dispatcher `open_calculator`)
- `tests/test_break.py::TestDispatcherBreak::test_lock_computer_without_cua` →
  `user32.LockWorkStation`
- `tests/test_win32_driver.py::TestWin32Driver::test_win32_driver_click_at_does_not_raise`
  → `user32.SetCursorPos` (fixed there; the win32gui focus calls above still
  leaked and were fixed this session in `2805785`)
- `tests/integration/test_planner_to_executor.py` `chain` fixture →
  `Win32Driver.click_at`

### Serial counts (verbatim)

- Before fix: `1 failed, 1135 passed, 1 skipped, 1 warning in 201.48s (0:03:21)`
- After fix: `1136 passed, 1 skipped, 1 warning in 200.28s (0:03:20)`
- Command: `./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rf`
  (serial). Matches PLAN §0's baseline of 1136 passed, 1 skipped.

Parallel run skipped: user runs it by hand, with their windows closed, after
reviewing the guard.

### Notes / left undone

- Step 2 of `incident-fix-gemini.md` is not started. The green run's
  `%TEMP%\grace-blocked-calls.txt` still lists two genuine leaks for it to
  chase (both pass today only because production code swallows the guard's
  `AssertionError`): `tests/test_offline_guarantee.py` (→ `os.startfile`) and
  `tests/test_verification_engine.py` (→ `win32gui.SetForegroundWindow`), plus
  the intentional `tests/test_desktop_guard.py` entries.
- Every full suite run appends two synthetic smoke-run lines
  (`"smoke_run": true`, model ids like `ch-test`) to each of
  `bench/results/fine_tune_*.jsonl` — tests write benchmark results into the
  repo instead of a temp dir. This session restored those four files to HEAD
  after verifying; the write target should be moved to temp in a follow-up.

### Process check

No Python, pytest or cargo process started by this session is still running.
Every command ran in the foreground and returned before reporting (both full
pytest runs exited, with codes 1 and 0). Verification command and output:

```
PS C:\Users\Ashman Das\Documents\Grace> Get-Process | Where-Object { $_.ProcessName -match '^(python|pytest|cargo|rustc|grace)' } | Format-Table Id, ProcessName, StartTime -AutoSize
(no output — no matching process exists)
```
