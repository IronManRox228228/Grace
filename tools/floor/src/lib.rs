//! Floor-machine simulation and throttle kit modules (PLAN §1, §2, §9).

use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum SafetyError {
    #[error("Safety guard: target directory '{0}' is not inside a designated temporary directory. Refusing to write to non-temp location.")]
    NonTempDirectory(String),
    #[error("Safety guard: requested RAM ({requested} MB) exceeds maximum safety ceiling ({ceiling} MB). Refusing to proceed.")]
    ExceedsRamCeiling { requested: usize, ceiling: usize },
    #[error("Invalid target bandwidth percentage ({0}). Must be between 1 and 95.")]
    InvalidBandwidthPercentage(u32),
}

/// Checks whether a given path is inside the system's temporary directory
/// or an explicitly designated `%TEMP%` / `%TMP%` directory.
pub fn is_temp_dir(path: &Path) -> bool {
    let canonical_target = match path.canonicalize() {
        Ok(c) => c,
        Err(_) => {
            // If target does not exist yet, check its existing parent
            let mut cur = path;
            while let Some(parent) = cur.parent() {
                if let Ok(c) = parent.canonicalize() {
                    return is_subpath_of_temp(&c);
                }
                cur = parent;
            }
            return false;
        }
    };
    is_subpath_of_temp(&canonical_target)
}

fn is_subpath_of_temp(path: &Path) -> bool {
    let mut temp_roots = Vec::new();

    // Standard env temp dir
    if let Ok(c) = std::env::temp_dir().canonicalize() {
        temp_roots.push(c);
    }
    // %TEMP%
    if let Ok(t) = std::env::var("TEMP") {
        if let Ok(c) = PathBuf::from(t).canonicalize() {
            temp_roots.push(c);
        }
    }
    // %TMP%
    if let Ok(t) = std::env::var("TMP") {
        if let Ok(c) = PathBuf::from(t).canonicalize() {
            temp_roots.push(c);
        }
    }

    let path_str = path.to_string_lossy().to_lowercase();
    for root in temp_roots {
        let root_str = root.to_string_lossy().to_lowercase();
        if path_str.starts_with(&root_str) {
            return true;
        }
    }

    false
}

/// Validates that a directory target for disk filling is safe.
pub fn validate_disk_fill_target(path: &Path) -> Result<(), SafetyError> {
    if !is_temp_dir(path) {
        return Err(SafetyError::NonTempDirectory(path.display().to_string()));
    }
    Ok(())
}

/// Validates that requested RAM allocation does not exceed the safety ceiling.
pub fn validate_ram_allocation(requested_mb: usize, ceiling_mb: usize) -> Result<(), SafetyError> {
    if requested_mb > ceiling_mb {
        return Err(SafetyError::ExceedsRamCeiling {
            requested: requested_mb,
            ceiling: ceiling_mb,
        });
    }
    Ok(())
}

/// Validates target memory bandwidth percentage.
pub fn validate_bandwidth_percentage(pct: u32) -> Result<(), SafetyError> {
    if pct == 0 || pct > 95 {
        return Err(SafetyError::InvalidBandwidthPercentage(pct));
    }
    Ok(())
}

/// Quick STREAM-style Triad benchmark:
/// A[i] = B[i] + scalar * C[i]
/// Returns measured memory bandwidth in GB/s.
pub fn measure_stream_bandwidth(buffer_elements: usize) -> f64 {
    let mut a = vec![0.0f64; buffer_elements];
    let b = vec![1.0f64; buffer_elements];
    let c = vec![2.0f64; buffer_elements];
    let scalar = 3.0f64;

    let start = std::time::Instant::now();
    for i in 0..buffer_elements {
        a[i] = b[i] + scalar * c[i];
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Prevent optimizer from eliminating calculation
    std::hint::black_box(&a);

    // STREAM Triad does 2 reads (B, C) and 1 write (A) per element
    // Total bytes = 3 * sizeof(f64) * buffer_elements = 24 * buffer_elements
    let total_bytes = 24.0 * buffer_elements as f64;
    (total_bytes / elapsed) / 1_000_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_disk_refuses_non_temp_dir() {
        let non_temp_paths = [
            PathBuf::from("C:\\"),
            PathBuf::from("C:\\Windows"),
            PathBuf::from("C:\\Program Files"),
            PathBuf::from("C:\\Users"),
        ];

        for path in &non_temp_paths {
            let res = validate_disk_fill_target(path);
            assert!(
                res.is_err(),
                "Path {} was expected to be rejected by safety check",
                path.display()
            );
            assert!(matches!(res.unwrap_err(), SafetyError::NonTempDirectory(_)));
        }
    }

    #[test]
    fn fill_disk_accepts_temp_dir() {
        let temp_dir = std::env::temp_dir();
        let target = temp_dir.join("grace_test_throttle_dir");
        assert!(
            validate_disk_fill_target(&target).is_ok(),
            "Path within temp dir should be accepted"
        );
    }

    #[test]
    fn clutter_refuses_exceeding_ram_ceiling() {
        assert_eq!(
            validate_ram_allocation(8192, 4096),
            Err(SafetyError::ExceedsRamCeiling {
                requested: 8192,
                ceiling: 4096,
            })
        );
    }

    #[test]
    fn clutter_accepts_within_ram_ceiling() {
        assert!(validate_ram_allocation(2048, 4096).is_ok());
        assert!(validate_ram_allocation(4096, 4096).is_ok());
    }

    #[test]
    fn bandwidth_hog_validates_target_pct() {
        assert!(validate_bandwidth_percentage(50).is_ok());
        assert!(validate_bandwidth_percentage(95).is_ok());
        assert_eq!(
            validate_bandwidth_percentage(0),
            Err(SafetyError::InvalidBandwidthPercentage(0))
        );
        assert_eq!(
            validate_bandwidth_percentage(100),
            Err(SafetyError::InvalidBandwidthPercentage(100))
        );
    }

    #[test]
    fn stream_bandwidth_measurement_computes_valid_throughput() {
        let gb_per_sec = measure_stream_bandwidth(100_000);
        assert!(gb_per_sec > 0.0);
    }
}
