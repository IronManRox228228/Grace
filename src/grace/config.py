import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Runtime settings, read from the environment at *construction* time.

    Every field is a `default_factory` rather than a plain default because a
    plain default evaluates once, when this module is imported - which for a
    frozen dataclass means `Config()` returns the environment as it stood at
    import, forever. Anything that loaded a `.env` afterwards, or set a variable
    for a test, was silently ignored, and the failure looks like the setting not
    working rather than the setting not being read.
    """

    # Model paths
    llama_model_path: str = field(default_factory=lambda: os.getenv(
        "LLAMA_MODEL_PATH",
        os.path.join(os.path.expanduser("~"), "Downloads", "UI-TARS-1.5-7B.Q4_K_M.gguf"),
    ))
    llama_mmproj_path: str = field(default_factory=lambda: os.getenv(
        "LLAMA_MMPROJ_PATH",
        os.path.join(os.path.expanduser("~"), "Downloads", "UI-TARS-1.5-7B.mmproj-Q8_0.gguf"),
    ))
    use_ui_tars_local: bool = field(default_factory=lambda: os.getenv("USE_UI_TARS_LOCAL", "true").lower() in ("true", "1", "yes"))
    # A separate local model for the planner role (JSON/tool-calling), used
    # only when running fully local. UI-TARS is grounding-tuned and a poor fit
    # for the planner's structured-output job, so this is deliberately a
    # different gguf from llama_model_path.
    local_planner_model_path: str = field(default_factory=lambda: os.getenv("LLAMA_PLANNER_MODEL_PATH", ""))
    kokoro_model_path: str = field(default_factory=lambda: os.getenv(
        "KOKORO_MODEL_PATH",
        os.path.join(
            os.path.expanduser("~"),
            ".cache", "huggingface", "hub", "models--hexgrad--Kokoro-82M",
            "snapshots", "f3ff3571791e39611d31c381e3a41a3af07b4987", "kokoro-v1_0.pth",
        ),
    ))
    kokoro_voices_path: str = field(default_factory=lambda: os.getenv(
        "KOKORO_VOICES_PATH",
        os.path.join(
            os.path.expanduser("~"),
            ".cache", "huggingface", "hub", "models--hexgrad--Kokoro-82M",
            "snapshots", "f3ff3571791e39611d31c381e3a41a3af07b4987", "voices", "af_bella.pt",
        ),
    ))

    # llama-server & Gemini Cloud LLM
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model_name: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite"))
    use_cloud_llm: bool = field(default_factory=lambda: os.getenv("USE_CLOUD_LLM", "true").lower() in ("true", "1", "yes"))

    llama_server_exe: str = field(default_factory=lambda: os.getenv(
        "LLAMA_SERVER_EXE",
        os.path.join(os.path.dirname(__file__), "..", "..", "llama cpp", "llama-server.exe"),
    ))
    llama_host: str = field(default_factory=lambda: os.getenv("LLAMA_HOST", "127.0.0.1"))
    llama_port: int = field(default_factory=lambda: int(os.getenv("LLAMA_PORT", "8080")))
    llama_context_window: int = field(default_factory=lambda: int(os.getenv("LLAMA_CONTEXT_WINDOW", "8192")))


    llama_ngl: int = field(default_factory=lambda: int(os.getenv("LLAMA_NGL", "999")))
    llama_cache_type_k: str = field(default_factory=lambda: os.getenv("LLAMA_CACHE_TYPE_K", "f16"))
    llama_cache_type_v: str = field(default_factory=lambda: os.getenv("LLAMA_CACHE_TYPE_V", "f16"))


    # Audio
    mic_device_index: int = field(default_factory=lambda: int(os.getenv("MIC_DEVICE_INDEX", "-1")))
    mic_sample_rate: int = field(default_factory=lambda: int(os.getenv("MIC_SAMPLE_RATE", "16000")))
    mic_chunk: int = field(default_factory=lambda: int(os.getenv("MIC_CHUNK", "512")))
    mic_channels: int = field(default_factory=lambda: int(os.getenv("MIC_CHANNELS", "1")))
    mic_width: int = field(default_factory=lambda: int(os.getenv("MIC_WIDTH", "2")))
    followup_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("FOLLOWUP_TIMEOUT_SECONDS", "10")))
    initial_listen_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("INITIAL_LISTEN_TIMEOUT_SECONDS", "6.0")))

    # Wake Word
    wake_word_keyword: str = field(default_factory=lambda: os.getenv("WAKE_WORD_KEYWORD", "grace"))
    wake_word_threshold: float = field(default_factory=lambda: float(os.getenv("WAKE_WORD_THRESHOLD", "0.8")))
    vosk_model_path: str = field(default_factory=lambda: os.getenv(
        "VOSK_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "vosk-model-small-en-us-0.15"),
    ))
    vosk_keyword: str = field(default_factory=lambda: os.getenv("VOSK_KEYWORD", "grace"))
    vosk_threshold: float = field(default_factory=lambda: float(os.getenv("VOSK_THRESHOLD", "0.4")))

    # Whisper
    whisper_model_path: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL_PATH", "small"))
    whisper_vad_threshold: float = field(default_factory=lambda: float(os.getenv("WHISPER_VAD_THRESHOLD", "0.008")))
    whisper_silence_duration_ms: int = field(default_factory=lambda: int(os.getenv("WHISPER_SILENCE_DURATION_MS", "1200")))

    # Kokoro
    # Two workers share ONE KModel: enough concurrency to hide synthesis behind
    # playback, without paying for a second copy of the model in VRAM.
    kokoro_workers: int = field(default_factory=lambda: int(os.getenv("KOKORO_WORKERS", "2")))
    kokoro_device: str = field(default_factory=lambda: os.getenv("KOKORO_DEVICE", "cuda"))
    # float16 is rejected by kokoro 0.9.4 (its internal tensors stay float32),
    # so float32 is the only working value; see kokoro_engine._resolve_dtype.
    kokoro_dtype: str = field(default_factory=lambda: os.getenv("KOKORO_DTYPE", "float32"))
    kokoro_warmup: bool = field(default_factory=lambda: os.getenv("KOKORO_WARMUP", "true").lower() in ("true", "1", "yes"))
    kokoro_cache_size: int = field(default_factory=lambda: int(os.getenv("KOKORO_CACHE_SIZE", "32")))
    kokoro_speed: float = field(default_factory=lambda: float(os.getenv("KOKORO_SPEED", "1.0")))

    # Agent loop. The step cap stays 0 = unlimited: capping steps only ever
    # stops a real task halfway ("I've used up my planning budget"), leaving the
    # desktop in a half-changed state, and a task that is making progress should
    # be allowed to take the steps it needs.
    agent_max_iterations: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_ITERATIONS", "0")))
    # Time and quota are the real ceilings, and unlike steps they bound the
    # failure mode rather than the task: a goal that is getting somewhere does
    # not spend three minutes or forty planner calls doing it, and one that is
    # not will spend both without ever tripping a step cap. This is what makes
    # the twelve-minute run structurally impossible rather than merely unlikely.
    agent_max_seconds: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_SECONDS", "180")))
    # The model the escalation ladder's third rung asks, after a step has been
    # tried as planned and then re-grounded without effect. Deliberately not the
    # configured planner: paying for capability on every step to have it
    # available on the rare one is the trade a tiered design exists to avoid.
    # Set empty to make that rung a plain retry with the failure evidence.
    stronger_planner_model: str = field(default_factory=lambda: os.getenv("STRONGER_PLANNER_MODEL", "gemini-3.1-flash"))
    planner_max_calls_per_goal: int = field(default_factory=lambda: int(os.getenv("PLANNER_MAX_CALLS_PER_GOAL", "40")))
    # Not unlimited, unlike the two above. Those bound a task that is making
    # progress; this bounds one that is making none, and "unlimited" there means
    # an unreachable LLM freezes Grace mid-task with no way to interrupt it.
    agent_max_consecutive_plan_failures: int = field(default_factory=lambda: int( os.getenv("AGENT_MAX_CONSECUTIVE_PLAN_FAILURES", "3")))
    # Likewise bounded: a planner retrying one action against one unchanging
    # view is not making progress, it is just spending quota.
    agent_max_repeated_actions: int = field(default_factory=lambda: int( os.getenv("AGENT_MAX_REPEATED_ACTIONS", "3")))
    screenshot_max_width: int = field(default_factory=lambda: int(os.getenv("SCREENSHOT_MAX_WIDTH", "1280")))
    # Fewest actionable controls a window can report and still be planned for
    # from its element list rather than from a marked screenshot. Raising it
    # sends more apps down the vision path: slower, and more likely to be right.
    observability_min_actionable: int = field(default_factory=lambda: int( os.getenv("OBSERVABILITY_MIN_ACTIONABLE", "8")))

    # Browser DOM access. Attach-only: Grace never launches a browser with a
    # debug flag and never touches the user's profile. Unset (0) = disabled,
    # in which case browser elements come from the UIA/ARIA tree instead.
    cdp_port: int = field(default_factory=lambda: int(os.getenv("GRACE_CDP_PORT", "0")))

    # OculiX / JPype visual fallback. Off by default: it starts a JVM in-process
    # and can block for seconds per unresolved click.
    use_oculix: bool = field(default_factory=lambda: os.getenv("USE_OCULIX", "false").lower() in ("true", "1", "yes"))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("GRACE_LOG_LEVEL", "INFO"))

    # WebSocket (frontend)
    ws_host: str = field(default_factory=lambda: os.getenv("WS_HOST", "127.0.0.1"))
    ws_port: int = field(default_factory=lambda: int(os.getenv("WS_PORT", "8765")))
    # Comma-separated origins to allow beyond the built-in Tauri/dev-server
    # allowlist (see ws_server._DEFAULT_ALLOWED_ORIGINS). Empty by default -
    # the server only needs this when the renderer is served from somewhere
    # non-standard.
    ws_allowed_origins: str = field(default_factory=lambda: os.getenv("WS_ALLOWED_ORIGINS", ""))

    # Derived
    @property
    def llama_server_url(self) -> str:
        return f"http://{self.llama_host}:{self.llama_port}"

    @property
    def model_swap_enabled(self) -> bool:
        """True when planner and grounder both need the same local GPU.

        Only relevant when the planner is local too (use_cloud_llm=false) and
        a distinct planner model has been configured. If the planner stays on
        Gemini, only UI-TARS ever needs the GPU and there is no contention.
        """
        return (not self.use_cloud_llm) and bool(self.local_planner_model_path)
