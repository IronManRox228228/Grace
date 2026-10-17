//! `bandwidth-hog`: memory bandwidth throttler.
//! Consumes a configurable share of memory bandwidth (streaming reads/writes over large buffers)
//! to approximate single-channel DDR4 performance; calibrates against a STREAM-style measurement;
//! stops cleanly on Ctrl+C.

use grace_floor::{measure_stream_bandwidth, validate_bandwidth_percentage};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static RUNNING: AtomicBool = AtomicBool::new(true);

#[cfg(windows)]
extern "system" fn console_ctrl_handler(_ctrl_type: u32) -> i32 {
    RUNNING.store(false, Ordering::SeqCst);
    1
}

fn register_ctrl_c() {
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn SetConsoleCtrlHandler(
                handler: Option<extern "system" fn(u32) -> i32>,
                add: i32,
            ) -> i32;
        }
        SetConsoleCtrlHandler(Some(console_ctrl_handler), 1);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut target_pct: u32 = 50;
    let mut threads: usize = (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) / 2).max(1);
    let mut buffer_mb: usize = 128;
    let mut calibrate_only = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--target-pct" => {
                if let Some(val) = args.next().and_then(|v| v.parse().ok()) {
                    target_pct = val;
                }
            }
            "--threads" => {
                if let Some(val) = args.next().and_then(|v| v.parse().ok()) {
                    threads = val;
                }
            }
            "--buffer-mb" => {
                if let Some(val) = args.next().and_then(|v| v.parse().ok()) {
                    buffer_mb = val;
                }
            }
            "--calibrate-only" => {
                calibrate_only = true;
            }
            "-h" | "--help" => {
                println!("bandwidth-hog: memory bandwidth throttling tool");
                println!("Usage: bandwidth-hog [--target-pct 50] [--threads N] [--buffer-mb 128] [--calibrate-only]");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    if let Err(e) = validate_bandwidth_percentage(target_pct) {
        eprintln!("Safety error: {e}");
        std::process::exit(1);
    }

    register_ctrl_c();

    println!("[bandwidth-hog] Calibrating system memory baseline (STREAM Triad)...");
    let baseline_gbps = measure_stream_bandwidth(2_000_000);
    println!("[bandwidth-hog] Baseline memory throughput: {:.2} GB/s", baseline_gbps);

    if calibrate_only {
        println!("[bandwidth-hog] Calibration complete.");
        return;
    }

    let target_burn_gbps = baseline_gbps * (target_pct as f64 / 100.0);
    println!(
        "[bandwidth-hog] Targeting {:.1}% bandwidth consumption (~{:.2} GB/s) across {} worker thread(s)...",
        target_pct, target_burn_gbps, threads
    );
    println!("[bandwidth-hog] Running throttle. Press Ctrl+C to terminate cleanly.");

    let mut handles = Vec::new();
    let elements_per_thread = (buffer_mb * 1024 * 1024) / (std::mem::size_of::<f64>() * 3);

    for _ in 0..threads {
        handles.push(std::thread::spawn(move || {
            let mut a = vec![0.0f64; elements_per_thread];
            let b = vec![1.0f64; elements_per_thread];
            let c = vec![2.0f64; elements_per_thread];
            let scalar = 1.0001f64;

            while RUNNING.load(Ordering::Relaxed) {
                for i in 0..elements_per_thread {
                    a[i] = b[i] + scalar * c[i];
                }
                std::hint::black_box(&a);
                // Sleep brief interval to control target duty cycle
                if target_pct < 90 {
                    let sleep_micros = (100 - target_pct) as u64 * 10;
                    std::thread::sleep(Duration::from_micros(sleep_micros));
                }
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    println!("\n[bandwidth-hog] Stopped cleanly.");
}
