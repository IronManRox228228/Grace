//! Ported from `src/grace/config.py`.
//!
//! The Python `Config` is a frozen dataclass whose fields are all
//! `default_factory` lambdas that read `os.environ` - deliberately, per that
//! file's own docstring, so a `.env` loaded after import (or a variable set
//! for a test) is still picked up. `Config::from_env` reproduces that: it is
//! a plain function call, not a `once_cell`/`lazy_static`, so constructing a
//! `Config` twice with different environments produces two different
//! configs, exactly like calling `Config()` twice in Python.

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"),
        Err(_) => default,
    }
}

fn env_int(key: &str, default: i64) -> i64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_float(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Runtime settings, read from the environment at *construction* time.
/// Field names and defaults are transcribed 1:1 from `src/grace/config.py`
/// so a `.env` written for the Python backend still configures this one.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    // --- Audio ---
    pub mic_device_index: i64,
    pub mic_sample_rate: i64,
    pub mic_chunk: i64,
    pub mic_channels: i64,
    pub mic_width: i64,
    pub followup_timeout_seconds: i64,
    pub initial_listen_timeout_seconds: f64,

    // --- Wake word ---
    pub wake_word_keyword: String,
    pub wake_word_threshold: f64,
    pub vosk_keyword: String,
    pub vosk_threshold: f64,

    // --- STT ---
    pub whisper_model_path: String,
    pub whisper_vad_threshold: f64,
    pub whisper_silence_duration_ms: i64,

    // --- TTS (Kokoro) ---
    pub kokoro_workers: i64,
    pub kokoro_device: String,
    pub kokoro_dtype: String,
    pub kokoro_warmup: bool,
    pub kokoro_cache_size: i64,
    pub kokoro_speed: f64,

    // --- LLM ---
    pub use_ui_tars_local: bool,
    pub local_planner_model_path: String,
    pub gemini_api_key: String,
    pub gemini_model_name: String,
    pub use_cloud_llm: bool,
    pub llama_host: String,
    pub llama_port: i64,
    pub llama_context_window: i64,
    pub llama_ngl: i64,

    // --- Agent loop / safety-relevant caps ---
    // Step cap stays 0 = unlimited on purpose (see config.py's comment):
    // capping steps only ever stops a real task halfway, leaving the desktop
    // in a half-changed state.
    pub agent_max_iterations: i64,
    pub agent_max_seconds: i64,
    pub stronger_planner_model: String,
    pub planner_max_calls_per_goal: i64,
    pub agent_max_consecutive_plan_failures: i64,
    pub agent_max_repeated_actions: i64,
    pub screenshot_max_width: i64,
    pub observability_min_actionable: i64,

    // --- Browser / vision extras ---
    pub cdp_port: i64,
    pub use_oculix: bool,

    // --- Logging ---
    pub log_level: String,

    // --- WebSocket (frontend) ---
    pub ws_host: String,
    pub ws_port: i64,
    pub ws_allowed_origins: String,
}

impl Config {
    /// Reads every field from the current process environment, exactly as
    /// `grace.config.Config()` does in Python.
    pub fn from_env() -> Self {
        Self {
            mic_device_index: env_int("MIC_DEVICE_INDEX", -1),
            mic_sample_rate: env_int("MIC_SAMPLE_RATE", 16000),
            mic_chunk: env_int("MIC_CHUNK", 512),
            mic_channels: env_int("MIC_CHANNELS", 1),
            mic_width: env_int("MIC_WIDTH", 2),
            followup_timeout_seconds: env_int("FOLLOWUP_TIMEOUT_SECONDS", 10),
            initial_listen_timeout_seconds: env_float("INITIAL_LISTEN_TIMEOUT_SECONDS", 6.0),
            wake_word_keyword: env_string("WAKE_WORD_KEYWORD", "grace"),
            wake_word_threshold: env_float("WAKE_WORD_THRESHOLD", 0.8),
            vosk_keyword: env_string("VOSK_KEYWORD", "grace"),
            vosk_threshold: env_float("VOSK_THRESHOLD", 0.4),
            whisper_model_path: env_string("WHISPER_MODEL_PATH", "small"),
            whisper_vad_threshold: env_float("WHISPER_VAD_THRESHOLD", 0.008),
            whisper_silence_duration_ms: env_int("WHISPER_SILENCE_DURATION_MS", 1200),
            kokoro_workers: env_int("KOKORO_WORKERS", 2),
            kokoro_device: env_string("KOKORO_DEVICE", "cuda"),
            kokoro_dtype: env_string("KOKORO_DTYPE", "float32"),
            kokoro_warmup: env_bool("KOKORO_WARMUP", true),
            kokoro_cache_size: env_int("KOKORO_CACHE_SIZE", 32),
            kokoro_speed: env_float("KOKORO_SPEED", 1.0),
            use_ui_tars_local: env_bool("USE_UI_TARS_LOCAL", true),
            local_planner_model_path: env_string("LLAMA_PLANNER_MODEL_PATH", ""),
            gemini_api_key: env_string("GEMINI_API_KEY", ""),
            gemini_model_name: env_string("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite"),
            use_cloud_llm: env_bool("USE_CLOUD_LLM", true),
            llama_host: env_string("LLAMA_HOST", "127.0.0.1"),
            llama_port: env_int("LLAMA_PORT", 8080),
            llama_context_window: env_int("LLAMA_CONTEXT_WINDOW", 8192),
            llama_ngl: env_int("LLAMA_NGL", 999),
            agent_max_iterations: env_int("AGENT_MAX_ITERATIONS", 0),
            agent_max_seconds: env_int("AGENT_MAX_SECONDS", 180),
            stronger_planner_model: env_string("STRONGER_PLANNER_MODEL", "gemini-3.1-flash"),
            planner_max_calls_per_goal: env_int("PLANNER_MAX_CALLS_PER_GOAL", 40),
            agent_max_consecutive_plan_failures: env_int(
                "AGENT_MAX_CONSECUTIVE_PLAN_FAILURES",
                3,
            ),
            agent_max_repeated_actions: env_int("AGENT_MAX_REPEATED_ACTIONS", 3),
            screenshot_max_width: env_int("SCREENSHOT_MAX_WIDTH", 1280),
            observability_min_actionable: env_int("OBSERVABILITY_MIN_ACTIONABLE", 8),
            cdp_port: env_int("GRACE_CDP_PORT", 0),
            use_oculix: env_bool("USE_OCULIX", false),
            log_level: env_string("GRACE_LOG_LEVEL", "INFO"),
            ws_host: env_string("WS_HOST", "127.0.0.1"),
            ws_port: env_int("WS_PORT", 8765),
            ws_allowed_origins: env_string("WS_ALLOWED_ORIGINS", ""),
        }
    }

    pub fn llama_server_url(&self) -> String {
        format!("http://{}:{}", self.llama_host, self.llama_port)
    }

    /// True when planner and grounder both need the same local GPU. Only
    /// relevant when the planner is local too (`use_cloud_llm=false`) and a
    /// distinct planner model has been configured.
    pub fn model_swap_enabled(&self) -> bool {
        !self.use_cloud_llm && !self.local_planner_model_path.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Config reads process-global env vars; serialise the tests that mutate
    // them so they cannot interleave (matches the spirit of pytest's
    // monkeypatch, which restores per-test).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn defaults_match_python_config_py() {
        let _guard = ENV_LOCK.lock().unwrap();
        for key in ["AGENT_MAX_SECONDS", "AGENT_MAX_ITERATIONS", "FOLLOWUP_TIMEOUT_SECONDS"] {
            std::env::remove_var(key);
        }
        let config = Config::from_env();
        assert_eq!(config.agent_max_seconds, 180);
        assert_eq!(config.agent_max_iterations, 0);
        assert_eq!(config.followup_timeout_seconds, 10);
        assert_eq!(config.llama_server_url(), "http://127.0.0.1:8080");
        assert!(!config.model_swap_enabled());
    }

    #[test]
    fn reads_the_environment_at_construction_not_at_first_use() {
        // Ported guard for the exact bug config.py's docstring documents: a
        // plain (non-factory) default only reads the environment once, at
        // import. `from_env()` must re-read every call.
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("AGENT_MAX_SECONDS", "42");
        let a = Config::from_env();
        assert_eq!(a.agent_max_seconds, 42);
        std::env::set_var("AGENT_MAX_SECONDS", "99");
        let b = Config::from_env();
        assert_eq!(b.agent_max_seconds, 99);
        std::env::remove_var("AGENT_MAX_SECONDS");
    }
}
