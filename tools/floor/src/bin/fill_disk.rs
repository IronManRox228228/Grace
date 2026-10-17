//! `fill-disk`: optional, opt-in SSD capacity throttler.
//! Writes only inside a chosen temp directory and deletes all files on exit.

use grace_floor::validate_disk_fill_target;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// RAII cleanup guard to ensure temp files are deleted even on panic or Ctrl+C.
struct CleanupGuard {
    files: Vec<PathBuf>,
}

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        println!("[fill-disk] Cleaning up temporary fill files...");
        for file in &self.files {
            if file.exists() {
                let _ = std::fs::remove_file(file);
            }
        }
        println!("[fill-disk] Cleanup complete.");
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut temp_dir: Option<PathBuf> = None;
    let mut fill_gb: usize = 5;
    let mut chunk_mb: usize = 64;
    let mut dry_run = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--temp-dir" => {
                temp_dir = args.next().map(PathBuf::from);
            }
            "--fill-gb" => {
                if let Some(val) = args.next().and_then(|v| v.parse().ok()) {
                    fill_gb = val;
                }
            }
            "--chunk-mb" => {
                if let Some(val) = args.next().and_then(|v| v.parse::<usize>().ok()) {
                    chunk_mb = val.max(1);
                }
            }
            "--dry-run" => {
                dry_run = true;
            }
            "-h" | "--help" => {
                println!("fill-disk: temporary SSD space consumption tool");
                println!("Usage: fill-disk --temp-dir <DIR> [--fill-gb 5] [--chunk-mb 64] [--dry-run]");
                println!("NOTE: target directory MUST reside within system temporary directories.");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    let target_dir = match temp_dir {
        Some(d) => d,
        None => std::env::temp_dir().join("grace_floor_disk_fill"),
    };

    if let Err(e) = validate_disk_fill_target(&target_dir) {
        eprintln!("Safety error: {e}");
        std::process::exit(1);
    }

    if dry_run {
        println!(
            "[fill-disk] Dry run: verified safe target directory at '{}'. Would allocate {} GB.",
            target_dir.display(),
            fill_gb
        );
        return;
    }

    if !target_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&target_dir) {
            eprintln!("Failed to create target directory {}: {e}", target_dir.display());
            std::process::exit(1);
        }
    }

    register_ctrl_c();

    let mut guard = CleanupGuard { files: Vec::new() };

    let chunk_bytes = chunk_mb * 1024 * 1024;
    let buffer = vec![0xAAu8; chunk_bytes];
    let total_chunks = (fill_gb * 1024) / chunk_mb;

    println!(
        "[fill-disk] Allocating {} GB in {} ({} chunks of {} MB)...",
        fill_gb,
        target_dir.display(),
        total_chunks,
        chunk_mb
    );

    for i in 0..total_chunks {
        if !RUNNING.load(Ordering::Relaxed) {
            break;
        }

        let file_path = target_dir.join(format!("grace_fill_{:04}.bin", i));
        match OpenOptions::new().write(true).create(true).truncate(true).open(&file_path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(&buffer) {
                    eprintln!("Warning: failed writing chunk {i}: {e}. Disk may be full.");
                    guard.files.push(file_path);
                    break;
                }
                guard.files.push(file_path);
            }
            Err(e) => {
                eprintln!("Warning: failed to create chunk file {i}: {e}");
                break;
            }
        }
    }

    println!("[fill-disk] Disk fill holding resident. Press Ctrl+C to clean up and exit.");
    while RUNNING.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}
