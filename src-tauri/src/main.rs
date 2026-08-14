// No console window in a release build - the overlay is the entire UI. Debug
// builds keep it, because it is where the backend's [Backend] lines land.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    grace_shell_lib::run()
}
