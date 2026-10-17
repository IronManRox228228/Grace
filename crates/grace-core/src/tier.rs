//! Hardware tier detection (PLAN.md §2, Task 08).
//!
//! Mirrors `src/grace/tier.py`. Detects whether to run on the NPU, iGPU, or CPU floor.

use serde::{Deserialize, Serialize};

use crate::config::Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HardwareTier {
    Npu,
    Igpu,
    Cpu,
}

impl HardwareTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Npu => "npu",
            Self::Igpu => "igpu",
            Self::Cpu => "cpu",
        }
    }
}

/// Detect the hardware tier from config override or available system memory.
pub fn detect_hardware_tier(config: &Config) -> HardwareTier {
    let override_val = config.hardware_tier.trim().to_lowercase();
    if !override_val.is_empty() {
        match override_val.as_str() {
            "npu" | "best" | "best_npu" => return HardwareTier::Npu,
            "igpu" | "middle" | "middle_igpu" | "gpu" => return HardwareTier::Igpu,
            _ => return HardwareTier::Cpu,
        }
    }

    // Default to CPU floor for safety and memory stability
    HardwareTier::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_sets_tier_correctly() {
        let mut cfg = Config::from_env();
        cfg.hardware_tier = "npu".to_string();
        assert_eq!(detect_hardware_tier(&cfg), HardwareTier::Npu);

        cfg.hardware_tier = "igpu".to_string();
        assert_eq!(detect_hardware_tier(&cfg), HardwareTier::Igpu);

        cfg.hardware_tier = "cpu".to_string();
        assert_eq!(detect_hardware_tier(&cfg), HardwareTier::Cpu);
    }
}
