//! Central thread budget and CPU core allocation (PLAN.md §4, Task 08).
//!
//! Mirrors `src/grace/thread_budget.py`.

use std::thread;

use crate::config::Config;
use crate::tier::HardwareTier;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadBudget {
    pub total_cores: usize,
    pub active_worker_threads: usize,
    pub llama_threads: usize,
    pub ort_intra_op_threads: usize,
    pub allow_tts_lookahead: bool,
}

pub fn calculate_thread_budget(tier: HardwareTier, config: &Config) -> ThreadBudget {
    let total_cores = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    if config.thread_budget > 0 {
        let active = (config.thread_budget as usize).min(total_cores);
        return ThreadBudget {
            total_cores,
            active_worker_threads: active,
            llama_threads: active,
            ort_intra_op_threads: active.min(4).max(1),
            allow_tts_lookahead: tier != HardwareTier::Cpu,
        };
    }

    let reserve = if total_cores >= 4 { 2 } else { 1 };
    let active_workers = total_cores.saturating_sub(reserve).max(1);

    match tier {
        HardwareTier::Cpu => ThreadBudget {
            total_cores,
            active_worker_threads: active_workers,
            llama_threads: active_workers.min(4),
            ort_intra_op_threads: active_workers.min(4),
            allow_tts_lookahead: false,
        },
        HardwareTier::Npu | HardwareTier::Igpu => ThreadBudget {
            total_cores,
            active_worker_threads: active_workers,
            llama_threads: active_workers.min(8),
            ort_intra_op_threads: active_workers.min(4),
            allow_tts_lookahead: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_floor_disables_lookahead() {
        let cfg = Config::from_env();
        let budget = calculate_thread_budget(HardwareTier::Cpu, &cfg);
        assert!(!budget.allow_tts_lookahead);
        assert!(budget.active_worker_threads >= 1);
        assert!(budget.llama_threads <= 4);
    }

    #[test]
    fn igpu_enables_lookahead() {
        let cfg = Config::from_env();
        let budget = calculate_thread_budget(HardwareTier::Igpu, &cfg);
        assert!(budget.allow_tts_lookahead);
    }
}
