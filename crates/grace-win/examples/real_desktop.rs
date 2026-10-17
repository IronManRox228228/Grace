//! Manual verification example for `grace-win` real desktop perception and actuation.
//!
//! **MANDATORY SAFETY NOTICE**:
//! - NEVER run this example in automated CI or test scripts (Hard Rule 1).
//! - It interacts with your live Windows desktop and foreground window.
//! - Run manually on a developer workstation:
//!   `cargo run -p grace-win --example real_desktop`
//!   or with actuation enabled:
//!   `cargo run -p grace-win --example real_desktop -- --actuate`

#[cfg(windows)]
fn main() {
    use grace_win::dpi::real::get_dpi_scale;
    use grace_win::screenshot::real::capture_foreground_window;
    use grace_win::win32::real::foreground_window;
    use grace_win::{
        DesktopActuator, ElementGraphSource, Win32Actuator, WindowsElementGraphSource,
        WindowsPerceptionSource, WindowsSystemActions,
    };
    use grace_core::dispatcher::SystemActions;
    use grace_core::perception::PerceptionSource;

    println!("============================================================");
    println!("grace-win Real Desktop Inspection & Actuation Manual Example");
    println!("============================================================");

    let args: Vec<String> = std::env::args().collect();
    let actuate = args.iter().any(|arg| arg == "--actuate");

    // 1. Foreground Window
    println!("\n[1] Querying Foreground Window...");
    match foreground_window() {
        Some(fg) => {
            println!("    Title: {}", fg.title);
            println!("    Class: {}", fg.class_name);
            println!("    HWND:  {:?}", fg.hwnd);
            println!("    Rect:  {:?}", fg.rect);
        }
        None => println!("    No foreground window found."),
    }

    // 2. DPI Scale
    let scale = get_dpi_scale(None);
    println!("\n[2] DPI Scale factor: {:.2}", scale);

    // 3. UIA Tree Walk
    println!("\n[3] Walking UIA Tree via WindowsElementGraphSource...");
    let mut uia_source = WindowsElementGraphSource::new();
    let start = std::time::Instant::now();
    match uia_source.snapshot() {
        Ok(snapshots) => {
            let elapsed = start.elapsed();
            println!("    Discovered {} elements in {:.2?}", snapshots.len(), elapsed);
            println!("    First 5 interactive / informative element summaries:");
            for (i, snap) in snapshots.iter().take(5).enumerate() {
                println!("      {}. {}", i + 1, snap.summary());
            }
        }
        Err(err) => println!("    UIA walk failed: {err}"),
    }

    // 4. Foreground Screenshot
    println!("\n[4] Capturing Foreground Window Screenshot (BitBlt -> PNG)...");
    match capture_foreground_window(None) {
        Ok((png_bytes, width, height)) => {
            println!("    Successfully captured screenshot: {} bytes ({}x{})", png_bytes.len(), width, height);
            if png_bytes.len() >= 8 && &png_bytes[0..8] == b"\x89PNG\r\n\x1a\n" {
                println!("    PNG magic header verified: valid image format.");
            } else {
                println!("    Warning: PNG magic header mismatch!");
            }
        }
        Err(err) => println!("    Screenshot failed: {err}"),
    }

    // 5. Perception Source Integration
    println!("\n[5] Testing WindowsPerceptionSource Integration...");
    let mut perception = WindowsPerceptionSource::new();
    let screen = perception.capture_snapshot();
    println!("    Screen snapshot window title: '{}'", screen.window_title);
    println!("    Screen dimensions: {}x{}", screen.width, screen.height);
    println!("    Captured {} element nodes.", screen.graph.elements.len());

    // 6. System Actions (safe read-only inspection)
    println!("\n[6] Testing WindowsSystemActions describe_screen...");
    let mut actions = WindowsSystemActions::new();
    let (desc_title, top_elems) = actions.describe_screen();
    println!("    Describe screen title: '{desc_title}'");
    println!("    Top elements: {:?}", top_elems);

    // 7. Actuator Check
    println!("\n[7] DesktopActuator Check:");
    if actuate {
        println!("    --actuate flag provided. Testing small mouse wheel scroll...");
        let mut actuator = Win32Actuator::new();
        match actuator.scroll(1) {
            Ok(()) => println!("    Successfully sent 1 scroll click via SendInput."),
            Err(e) => println!("    Actuation error: {e}"),
        }
    } else {
        println!("    Actuation skipped (pass --actuate to test mouse wheel scroll).");
    }

    println!("\nReal desktop manual verification completed successfully.\n");
}

#[cfg(not(windows))]
fn main() {
    println!("grace-win real desktop example is only supported on Windows.");
}
