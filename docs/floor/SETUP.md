# Floor-Machine Setup & Hardware Simulation Guide

This document describes how to configure any Windows machine to physically match the Grace CPU Floor Machine specification (PLAN.md §1, §2, §9), and how to use the software throttle kit (`tools/floor/`) when hardware cannot be altered.

---

## 1. Target Floor Machine Specification

The Grace floor specification defines the minimum hardware Grace must run reliably on:
- **Processor:** Intel Core i5-8300H (4 cores, 8 threads @ 2.3 GHz base, 4.0 GHz boost)
- **Memory:** 8 GB single-channel DDR4-2400 (or dual-channel restricted to single-channel bandwidth)
- **Graphics:** Integrated Intel UHD Graphics 630 (dedicated dGPU disabled)
- **Storage:** Nearly full 128 GB SATA SSD (~10 GB free)
- **Background load:** "20 Chrome tabs" holding resident memory and generating CPU/memory churn

---

## 2. Manual Hardware & OS Configuration Steps

> [!NOTE]
> All steps in this section are completely reversible. Follow the "How to Undo" instructions after benchmarking to restore full system capabilities.

### 2.1 Restricting System RAM to 8192 MB (`msconfig`)

This simulates an 8 GB RAM floor environment on a system with 16 GB, 32 GB, or 64 GB RAM.

#### Setup:
1. Press `Win + R`, type `msconfig`, and press **Enter** (requires Administrator privileges).
2. Navigate to the **Boot** tab.
3. Select your current Windows installation and click **Advanced options...**.
4. Check the **Maximum memory** box.
5. Enter `8192` (representing 8192 MB / 8 GB).
6. Click **OK**, then click **Apply** and **OK**.
7. Windows will prompt you to restart. Click **Restart**.
8. After rebooting, open Task Manager (`Ctrl + Shift + Esc`) → **Performance** → **Memory** and confirm total physical memory reads as approximately `7.9 GB` to `8.0 GB`.

#### How to Undo:
1. Press `Win + R`, type `msconfig`, and press **Enter**.
2. Go to **Boot** → **Advanced options...**.
3. **Uncheck** the **Maximum memory** checkbox.
4. Click **OK**, **Apply**, and restart your computer. Total RAM will revert to full capacity.

---

### 2.2 Disabling Dedicated GPU in Device Manager

This forces all rendering, ONNX, and LLM inference onto the CPU / Intel iGPU, ensuring dGPU acceleration is not accidentally utilized.

#### Setup:
1. Press `Win + X` and select **Device Manager** (or run `devmgmt.msc`).
2. Expand the **Display adapters** category.
3. Locate your dedicated graphics card (e.g., `NVIDIA GeForce GTX 1050`, `RTX 4060`, `AMD Radeon RX`).
4. Right-click the dedicated adapter and select **Disable device**.
5. Click **Yes** on the confirmation prompt.
6. Verify that only the integrated GPU (`Intel(R) UHD Graphics 630` or similar) remains active.
7. Open Task Manager (`Ctrl + Shift + Esc`) → **Performance** tab and confirm `GPU 1 (NVIDIA)` is no longer active.

#### How to Undo:
1. Open **Device Manager** (`devmgmt.msc`).
2. Expand **Display adapters**.
3. Right-click the disabled dedicated GPU and select **Enable device**.
4. The graphics card is immediately re-enabled.

---

### 2.3 Setting Windows Power Plan

This prevents high-wattage desktop turbo boosting that masks thermal throttling on mobile CPU floor hardware.

#### Setup:
1. Press `Win + R`, type `powercfg.cpl`, and press **Enter**.
2. Select the **Balanced** power plan (recommended).
3. Click **Change plan settings** → **Change advanced power settings**.
4. Expand **Processor power management** → **Maximum processor state**:
   - Set to `99%` on both battery and plugged-in (disables Intel Turbo Boost, keeping frequencies at base ~2.3 GHz).
5. Click **Apply** and **OK**.

#### How to Undo:
1. Open `powercfg.cpl`.
2. Click **Change plan settings** → **Restore default settings for this plan**, or set Maximum processor state back to `100%`.

---

## 3. Software Throttle Kit (`tools/floor/`)

When borrowing hardware where `msconfig` reboots or dGPU toggles are restricted, the Grace Throttle Kit simulates floor constraints in software.

### 3.1 Memory Bandwidth Throttler (`bandwidth-hog`)
Simulates the reduced memory throughput of single-channel DDR4 memory by streaming read/writes across memory buffers.
```cmd
# 1. Run baseline calibration:
cargo run -p grace-floor --bin bandwidth-hog -- --calibrate-only

# 2. Run bandwidth throttle (e.g. consuming 50% bandwidth):
cargo run -p grace-floor --bin bandwidth-hog -- --target-pct 50
```
- Stops cleanly on `Ctrl+C`.

### 3.2 Background Load & RAM Clutter (`clutter`)
Commits a specified amount of resident RAM and runs a light, periodic background dirtying loop mimicking 20 open Chrome tabs:
```cmd
# Holds 2048 MB resident across 20 simulated tabs:
cargo run -p grace-floor --bin clutter -- --ram-mb 2048 --tabs 20
```
- Enforces safety ceiling (default 6144 MB) to prevent OS out-of-memory panics.
- Stops cleanly and frees all memory on `Ctrl+C`.

### 3.3 Disk Capacity Simulator (`fill-disk`)
Opt-in tool to fill disk space up to a threshold inside designated temporary folders:
```cmd
# Verifies safe target directory:
cargo run -p grace-floor --bin fill-disk -- --dry-run

# Creates 5 GB of temp chunks in %TEMP%:
cargo run -p grace-floor --bin fill-disk -- --fill-gb 5
```
- **Safety guarantee:** Refuses to write outside `%TEMP%` / `%TMP%`.
- Automatically deletes all created fill files on `Ctrl+C` or exit.

---

## 4. One-Command Floor Benchmark Run

To run the complete benchmark suite on the floor setup:
```powershell
powershell -File tools/floor/run_floor_benchmarks.ps1
```
This runs the latency tests (§9: wake-to-earcon, pattern command, one-step model decision, multi-step progress), measures timings, and writes a consolidated report to `docs/benchmarks/floor.md`.
