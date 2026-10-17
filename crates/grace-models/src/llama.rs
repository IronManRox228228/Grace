//! Management of local `llama-server` process lifecycle, hardware presets,
//! slot configurations, and Job Object supervision.
//!
//! Provides:
//! - `LlamaServerConfig`: CLI argument builder for llama-server.
//! - `LlamaServerManager`: process supervisor with health checks, restarts,
//!   and Windows Job Object assignment with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
//!   (task 04b) so child processes are never orphaned.
//! - `FleetConfig` & `SlotConfig`: per-slot model configurations and presets
//!   (`cpu-fleet` and `gpu-single`, task 13).

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// Preset tier for local llama-server serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preset {
    /// Micromodel fleet on CPU floor: zero GPU layers, q8 cache, 2048 ctx.
    #[default]
    CpuFleet,
    /// Buildathon single-model setup: 999 GPU layers, f16 cache, 8192 ctx.
    GpuSingle,
    /// Custom user-specified configuration.
    Custom,
}

impl Preset {
    pub fn as_str(&self) -> &'static str {
        match self {
            Preset::CpuFleet => "cpu-fleet",
            Preset::GpuSingle => "gpu-single",
            Preset::Custom => "custom",
        }
    }
}

/// Configuration for a specific micromodel slot (task 13).
#[derive(Debug, Clone, PartialEq)]
pub struct SlotConfig {
    pub slot_name: String,
    pub model_name: String,
    pub model_path: PathBuf,
    pub runtime: String, // "llama_cpp", "onnx", "ort", "transformers", "internal"
    pub threshold: f32,
    pub margin: f32,
    pub context_window: u32,
    pub temperature: f32,
    pub extra: serde_json::Value,
}

/// Collection of all slots for a given preset ("gpu-single" vs "cpu-fleet").
#[derive(Debug, Clone, PartialEq)]
pub struct FleetConfig {
    pub preset: Preset,
    pub slots: HashMap<String, SlotConfig>,
}

impl FleetConfig {
    pub fn get_slot(&self, name: &str) -> Option<&SlotConfig> {
        self.slots.get(name)
    }

    pub fn is_fleet(&self) -> bool {
        self.preset == Preset::CpuFleet
    }

    pub fn is_single(&self) -> bool {
        self.preset == Preset::GpuSingle
    }

    /// Buildathon single-model preset: Holo3.1-4B (or configured GGUF) runs all slots.
    pub fn build_gpu_single_preset(model_path: Option<&Path>, mmproj_path: Option<&Path>) -> Self {
        let default_model = model_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models/Holo-3.1-4B.Q4_K_M.gguf"));
        let default_mmproj = mmproj_path.map(PathBuf::from);

        let common_slot = |name: &str| SlotConfig {
            slot_name: name.to_string(),
            model_name: "Holo-3.1-4B".to_string(),
            model_path: default_model.clone(),
            runtime: "llama_cpp".to_string(),
            threshold: 0.50,
            margin: 0.10,
            context_window: 8192,
            temperature: 0.0,
            extra: serde_json::json!({
                "mmproj": default_mmproj.as_ref().map(|p| p.to_string_lossy().to_string())
            }),
        };

        let mut slots = HashMap::new();
        for slot in ["tool_calling", "element_selection", "yes_no", "routing", "vision", "midsize_fallback"] {
            slots.insert(slot.to_string(), common_slot(slot));
        }

        Self {
            preset: Preset::GpuSingle,
            slots,
        }
    }

    /// Micromodel fleet preset for the CPU floor and edge environments (task 13).
    pub fn build_cpu_fleet_preset(models_dir: Option<&Path>) -> Self {
        let base_dir = models_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models"));
        let fleet_dir = base_dir.join("fleet");

        let mut slots = HashMap::new();
        slots.insert(
            "tool_calling".to_string(),
            SlotConfig {
                slot_name: "tool_calling".to_string(),
                model_name: "LFM2.5-350M-ToolCaller".to_string(),
                model_path: fleet_dir.join("lfm-350m-tool-caller.gguf"),
                runtime: "llama_cpp".to_string(),
                threshold: 0.75,
                margin: 0.15,
                context_window: 2048,
                temperature: 0.0,
                extra: serde_json::json!({"challenger": "FunctionGemma-270M"}),
            },
        );
        slots.insert(
            "element_selection".to_string(),
            SlotConfig {
                slot_name: "element_selection".to_string(),
                model_name: "GLiNER2.5-Decide-340M".to_string(),
                model_path: fleet_dir.join("gliner-decide-340m.onnx"),
                runtime: "onnx".to_string(),
                threshold: 0.70,
                margin: 0.15,
                context_window: 1024,
                temperature: 0.0,
                extra: serde_json::Value::Null,
            },
        );
        slots.insert(
            "yes_no".to_string(),
            SlotConfig {
                slot_name: "yes_no".to_string(),
                model_name: "GLiNER2.5-Decide-340M".to_string(),
                model_path: fleet_dir.join("gliner-decide-340m.onnx"),
                runtime: "onnx".to_string(),
                threshold: 0.65,
                margin: 0.10,
                context_window: 512,
                temperature: 0.0,
                extra: serde_json::Value::Null,
            },
        );
        slots.insert(
            "routing".to_string(),
            SlotConfig {
                slot_name: "routing".to_string(),
                model_name: "Contrastive-Router".to_string(),
                model_path: fleet_dir.join("contrastive_head.onnx"),
                runtime: "onnx".to_string(),
                threshold: 0.80,
                margin: 0.20,
                context_window: 512,
                temperature: 0.0,
                extra: serde_json::Value::Null,
            },
        );
        slots.insert(
            "vision".to_string(),
            SlotConfig {
                slot_name: "vision".to_string(),
                model_name: "LFM2.5-VL-450M".to_string(),
                model_path: fleet_dir.join("lfm-vl-450m.gguf"),
                runtime: "llama_cpp".to_string(),
                threshold: 0.60,
                margin: 0.10,
                context_window: 4096,
                temperature: 0.0,
                extra: serde_json::Value::Null,
            },
        );
        slots.insert(
            "midsize_fallback".to_string(),
            SlotConfig {
                slot_name: "midsize_fallback".to_string(),
                model_name: "LFM2.5-1.2B-Instruct".to_string(),
                model_path: fleet_dir.join("lfm-1.2b-instruct.gguf"),
                runtime: "llama_cpp".to_string(),
                threshold: 0.65,
                margin: 0.10,
                context_window: 4096,
                temperature: 0.0,
                extra: serde_json::json!({"on_demand_only": true}),
            },
        );

        Self {
            preset: Preset::CpuFleet,
            slots,
        }
    }
}

/// Configuration for launching `llama-server`.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaServerConfig {
    pub exe_path: PathBuf,
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
    pub host: String,
    pub port: u16,
    pub ngl: u32,
    pub context_window: u32,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub threads: Option<u32>,
    pub slots: Option<u32>,
    pub health_timeout: Duration,
    pub preset: Preset,
}

impl Default for LlamaServerConfig {
    fn default() -> Self {
        Self {
            exe_path: PathBuf::from("llama-server.exe"),
            model_path: PathBuf::new(),
            mmproj_path: None,
            host: "127.0.0.1".to_string(),
            port: 8080,
            ngl: 0,
            context_window: 2048,
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "q8_0".to_string(),
            threads: None,
            slots: None,
            health_timeout: Duration::from_secs(120),
            preset: Preset::CpuFleet,
        }
    }
}

impl LlamaServerConfig {
    /// Constructs a configuration from a preset.
    pub fn from_preset(preset: Preset, exe_path: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        match preset {
            Preset::CpuFleet => Self {
                exe_path: exe_path.into(),
                model_path: model_path.into(),
                mmproj_path: None,
                host: "127.0.0.1".to_string(),
                port: 8080,
                ngl: 0,
                context_window: 2048,
                cache_type_k: "q8_0".to_string(),
                cache_type_v: "q8_0".to_string(),
                threads: None,
                slots: None,
                health_timeout: Duration::from_secs(120),
                preset: Preset::CpuFleet,
            },
            Preset::GpuSingle => Self {
                exe_path: exe_path.into(),
                model_path: model_path.into(),
                mmproj_path: None,
                host: "127.0.0.1".to_string(),
                port: 8080,
                ngl: 999,
                context_window: 8192,
                cache_type_k: "f16".to_string(),
                cache_type_v: "f16".to_string(),
                threads: None,
                slots: None,
                health_timeout: Duration::from_secs(120),
                preset: Preset::GpuSingle,
            },
            Preset::Custom => Self {
                exe_path: exe_path.into(),
                model_path: model_path.into(),
                preset: Preset::Custom,
                ..Default::default()
            },
        }
    }

    /// Sets the multimodal projector path (`--mmproj`).
    pub fn with_mmproj(mut self, mmproj: impl Into<PathBuf>) -> Self {
        self.mmproj_path = Some(mmproj.into());
        self
    }

    /// Generates CLI arguments for `llama-server`.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = vec![
            "--host".to_string(),
            self.host.clone(),
            "--port".to_string(),
            self.port.to_string(),
            "-m".to_string(),
            self.model_path.to_string_lossy().to_string(),
            "-c".to_string(),
            self.context_window.to_string(),
            "-ngl".to_string(),
            self.ngl.to_string(),
            "--cache-type-k".to_string(),
            self.cache_type_k.clone(),
            "--cache-type-v".to_string(),
            self.cache_type_v.clone(),
        ];

        if let Some(mmproj) = &self.mmproj_path {
            args.push("--mmproj".to_string());
            args.push(mmproj.to_string_lossy().to_string());
        }

        if let Some(threads) = self.threads {
            args.push("-t".to_string());
            args.push(threads.to_string());
        }

        if let Some(slots) = self.slots {
            args.push("-np".to_string());
            args.push(slots.to_string());
        }

        args
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

/// Manages a running `llama-server` process under a Windows Job Object (task 04b).
pub struct LlamaServerManager {
    config: LlamaServerConfig,
    child: Option<Child>,
    #[cfg(windows)]
    job_handle: Option<windows::Win32::Foundation::HANDLE>,
    client: reqwest::blocking::Client,
}

impl LlamaServerManager {
    pub fn new(config: LlamaServerConfig) -> Self {
        Self {
            config,
            child: None,
            #[cfg(windows)]
            job_handle: None,
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
        }
    }

    pub fn config(&self) -> &LlamaServerConfig {
        &self.config
    }

    pub fn is_running(&mut self) -> bool {
        if let Some(child) = &mut self.child {
            match child.try_wait() {
                Ok(None) => true,
                _ => {
                    self.child = None;
                    false
                }
            }
        } else {
            false
        }
    }

    /// Starts the `llama-server` process and assigns it to a Job Object on Windows.
    pub fn start(&mut self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        let args = self.config.to_args();
        let mut cmd = Command::new(&self.config.exe_path);
        cmd.args(&args);
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn llama-server at {}", self.config.exe_path.display()))?;

        #[cfg(windows)]
        unsafe {
            use std::os::windows::io::AsRawHandle;
            use windows::Win32::Foundation::HANDLE;
            use windows::Win32::System::JobObjects::*;

            if let Ok(job) = CreateJobObjectW(None, None) {
                let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let _ = SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                let _ = AssignProcessToJobObject(job, HANDLE(child.as_raw_handle() as _));
                self.job_handle = Some(job);
            }
        }

        self.child = Some(child);
        Ok(())
    }

    /// Stops the `llama-server` process. On Windows, uses taskkill to kill the entire tree.
    pub fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                let pid = child.id();
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .creation_flags(0x08000000) // CREATE_NO_WINDOW
                    .status();
            }
            let _ = child.kill();
            let _ = child.wait();
        }

        #[cfg(windows)]
        if let Some(job) = self.job_handle.take() {
            unsafe {
                use windows::Win32::Foundation::CloseHandle;
                let _ = CloseHandle(job);
            }
        }

        Ok(())
    }

    /// Restarts the `llama-server` process.
    pub fn restart(&mut self) -> Result<()> {
        self.stop()?;
        self.start()
    }

    /// Health check: checks if `llama-server` returns HTTP 200 on `/health`.
    pub fn health_check(&self) -> bool {
        let url = format!("{}/health", self.config.base_url());
        match self.client.get(&url).send() {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Waits until the server is ready to accept completions, checking `/health` and testing a tiny completion.
    pub fn wait_until_ready(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if self.health_check() {
                // Test a 1-token completion
                let url = format!("{}/v1/chat/completions", self.config.base_url());
                let payload = serde_json::json!({
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": false
                });
                if let Ok(resp) = self.client.post(&url).json(&payload).send() {
                    if resp.status().is_success() {
                        return true;
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        false
    }
}

impl Drop for LlamaServerManager {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_fleet_preset_generates_expected_cli_args() {
        let config = LlamaServerConfig::from_preset(
            Preset::CpuFleet,
            "C:\\llama\\llama-server.exe",
            "C:\\models\\lfm-350m.gguf",
        );
        let args = config.to_args();
        assert_eq!(args[0], "--host");
        assert_eq!(args[1], "127.0.0.1");
        assert_eq!(args[2], "--port");
        assert_eq!(args[3], "8080");
        assert_eq!(args[4], "-m");
        assert_eq!(args[5], "C:\\models\\lfm-350m.gguf");
        assert_eq!(args[6], "-c");
        assert_eq!(args[7], "2048");
        assert_eq!(args[8], "-ngl");
        assert_eq!(args[9], "0");
        assert_eq!(args[10], "--cache-type-k");
        assert_eq!(args[11], "q8_0");
        assert_eq!(args[12], "--cache-type-v");
        assert_eq!(args[13], "q8_0");
    }

    #[test]
    fn gpu_single_preset_with_mmproj_generates_flags() {
        let config = LlamaServerConfig::from_preset(
            Preset::GpuSingle,
            "C:\\llama\\llama-server.exe",
            "C:\\models\\Holo-3.1-4B.gguf",
        )
        .with_mmproj("C:\\models\\Holo-mmproj.gguf");

        let args = config.to_args();
        assert!(args.contains(&"-ngl".to_string()));
        let ngl_idx = args.iter().position(|a| a == "-ngl").unwrap();
        assert_eq!(args[ngl_idx + 1], "999");
        assert!(args.contains(&"--mmproj".to_string()));
        let mm_idx = args.iter().position(|a| a == "--mmproj").unwrap();
        assert_eq!(args[mm_idx + 1], "C:\\models\\Holo-mmproj.gguf");
    }

    #[test]
    fn fleet_config_presets_define_required_slots() {
        let fleet = FleetConfig::build_cpu_fleet_preset(None);
        assert!(fleet.is_fleet());
        assert!(!fleet.is_single());
        assert!(fleet.get_slot("tool_calling").is_some());
        assert!(fleet.get_slot("element_selection").is_some());
        assert!(fleet.get_slot("yes_no").is_some());
        assert!(fleet.get_slot("routing").is_some());
        assert!(fleet.get_slot("vision").is_some());
        assert!(fleet.get_slot("midsize_fallback").is_some());

        let gpu = FleetConfig::build_gpu_single_preset(None, None);
        assert!(!gpu.is_fleet());
        assert!(gpu.is_single());
        assert_eq!(gpu.get_slot("tool_calling").unwrap().model_name, "Holo-3.1-4B");
    }

    #[test]
    fn manager_is_not_running_before_start() {
        let config = LlamaServerConfig::default();
        let mut manager = LlamaServerManager::new(config);
        assert!(!manager.is_running());
    }
}
