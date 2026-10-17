//! `clutter`: holds a configurable amount of RAM resident and simulates
//! "20 Chrome tabs" of background CPU/RAM churn, without opening a browser.

use grace_floor::validate_ram_allocation;
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
    let mut ram_mb: usize = 2048;
    let mut max_ram_mb: usize = 6144; // Default safety ceiling: 6 GB
    let mut tabs: usize = 20;
    let mut churn_interval_ms: u64 = 1000;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ram-mb" => {
                if let Some(val) = args.next().and_then(|v| v.parse().ok()) {
                    ram_mb = val;
                }
            }
            "--max-ram-mb" => {
                if let Some(val) = args.next().and_then(|v| v.parse().ok()) {
                    max_ram_mb = val;
                }
            }
            "--tabs" => {
                if let Some(val) = args.next().and_then(|v| v.parse::<usize>().ok()) {
                    tabs = val.max(1);
                }
            }
            "--churn-interval-ms" => {
                if let Some(val) = args.next().and_then(|v| v.parse().ok()) {
                    churn_interval_ms = val;
                }
            }
            "-h" | "--help" => {
                println!("clutter: background memory & CPU churn simulator");
                println!("Usage: clutter [--ram-mb 2048] [--max-ram-mb 6144] [--tabs 20] [--churn-interval-ms 1000]");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    if let Err(e) = validate_ram_allocation(ram_mb, max_ram_mb) {
        eprintln!("Safety error: {e}");
        std::process::exit(1);
    }

    register_ctrl_c();

    println!("[clutter] Allocating and committing {} MB RAM across {} simulated tabs...", ram_mb, tabs);
    let bytes_per_tab = (ram_mb * 1024 * 1024) / tabs;
    let mut tab_buffers: Vec<Vec<u8>> = Vec::with_capacity(tabs);

    for i in 0..tabs {
        let mut buf = vec![0u8; bytes_per_tab];
        // Touch every 4KB memory page to ensure physical commitment
        for page in buf.chunks_mut(4096) {
            page[0] = ((i * 37) % 255) as u8;
        }
        tab_buffers.push(buf);
    }
    println!("[clutter] Successfully committed {} MB resident memory.", ram_mb);
    println!("[clutter] Simulating background DOM/JS churn every {}ms. Press Ctrl+C to stop.", churn_interval_ms);

    let mut cycle = 0;
    while RUNNING.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(churn_interval_ms));
        if !RUNNING.load(Ordering::Relaxed) {
            break;
        }

        cycle = (cycle + 1) % tabs;
        // Churn: dirties memory in active tab buffer and performs slight CPU work
        let buf = &mut tab_buffers[cycle];
        for chunk in buf.chunks_mut(16384) {
            chunk[0] = chunk[0].wrapping_add(1);
        }
    }

    println!("\n[clutter] Releasing allocated memory and shutting down cleanly.");
    drop(tab_buffers);
}
