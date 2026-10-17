//! Process execution priority and EcoQoS management (PLAN.md §4, Task 08).
//!
//! Mirrors `src/grace/priority.py`.

#[cfg(windows)]
mod win {
    type HANDLE = *mut std::ffi::c_void;
    type DWORD = u32;
    type BOOL = i32;

    const ABOVE_NORMAL_PRIORITY_CLASS: DWORD = 0x00008000;
    const BELOW_NORMAL_PRIORITY_CLASS: DWORD = 0x00004000;
    const NORMAL_PRIORITY_CLASS: DWORD = 0x00000020;

    extern "system" {
        fn GetCurrentProcess() -> HANDLE;
        fn SetPriorityClass(hProcess: HANDLE, dwPriorityClass: DWORD) -> BOOL;
    }

    pub fn set_turn_priority() -> bool {
        unsafe {
            let proc = GetCurrentProcess();
            SetPriorityClass(proc, ABOVE_NORMAL_PRIORITY_CLASS) != 0
        }
    }

    pub fn set_idle_priority() -> bool {
        unsafe {
            let proc = GetCurrentProcess();
            SetPriorityClass(proc, NORMAL_PRIORITY_CLASS) != 0
        }
    }

    pub fn set_background_priority() -> bool {
        unsafe {
            let proc = GetCurrentProcess();
            SetPriorityClass(proc, BELOW_NORMAL_PRIORITY_CLASS) != 0
        }
    }
}

#[cfg(not(windows))]
mod fallback {
    pub fn set_turn_priority() -> bool {
        false
    }
    pub fn set_idle_priority() -> bool {
        false
    }
    pub fn set_background_priority() -> bool {
        false
    }
}

#[cfg(windows)]
pub use win::*;

#[cfg(not(windows))]
pub use fallback::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_calls_do_not_panic() {
        let _ = set_turn_priority();
        let _ = set_idle_priority();
        let _ = set_background_priority();
        let _ = set_idle_priority();
    }
}
