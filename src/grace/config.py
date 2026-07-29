import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Model paths
    llama_model_path: str = os.getenv(
        "LLAMA_MODEL_PATH",
        r"C:\Users\Ashman Das\Downloads\UI-TARS-1.5-7B.Q4_K_M.gguf",
    )
    llama_mmproj_path: str = os.getenv(
        "LLAMA_MMPROJ_PATH",
        r"C:\Users\Ashman Das\Downloads\UI-TARS-1.5-7B.mmproj-Q8_0.gguf",
    )
    use_ui_tars_local: bool = os.getenv("USE_UI_TARS_LOCAL", "true").lower() in ("true", "1", "yes")
    kokoro_model_path: str = os.getenv(
        "KOKORO_MODEL_PATH",
        r"C:\Users\Ashman Das\.cache\huggingface\hub\models--hexgrad--Kokoro-82M\snapshots\f3ff3571791e39611d31c381e3a41a3af07b4987\kokoro-v1_0.pth",
    )
    kokoro_voices_path: str = os.getenv(
        "KOKORO_VOICES_PATH",
        r"C:\Users\Ashman Das\.cache\huggingface\hub\models--hexgrad--Kokoro-82M\snapshots\f3ff3571791e39611d31c381e3a41a3af07b4987\voices\af_bella.pt",
    )

    # llama-server & Gemini Cloud LLM
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
    use_cloud_llm: bool = os.getenv("USE_CLOUD_LLM", "true").lower() in ("true", "1", "yes")

    llama_server_exe: str = os.getenv(
        "LLAMA_SERVER_EXE",
        os.path.join(os.path.dirname(__file__), "..", "..", "llama cpp", "llama-server.exe"),
    )
    llama_host: str = os.getenv("LLAMA_HOST", "127.0.0.1")
    llama_port: int = int(os.getenv("LLAMA_PORT", "8080"))
    llama_context_window: int = int(os.getenv("LLAMA_CONTEXT_WINDOW", "8192"))


    llama_ngl: int = int(os.getenv("LLAMA_NGL", "999"))
    llama_cache_type_k: str = os.getenv("LLAMA_CACHE_TYPE_K", "f16")
    llama_cache_type_v: str = os.getenv("LLAMA_CACHE_TYPE_V", "f16")


    # Audio
    mic_device_index: int = int(os.getenv("MIC_DEVICE_INDEX", "-1"))
    mic_sample_rate: int = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
    mic_chunk: int = int(os.getenv("MIC_CHUNK", "512"))
    mic_channels: int = int(os.getenv("MIC_CHANNELS", "1"))
    mic_width: int = int(os.getenv("MIC_WIDTH", "2"))
    followup_timeout_seconds: int = int(os.getenv("FOLLOWUP_TIMEOUT_SECONDS", "10"))

    # Wake Word
    wake_word_keyword: str = os.getenv("WAKE_WORD_KEYWORD", "grace")
    wake_word_threshold: float = float(os.getenv("WAKE_WORD_THRESHOLD", "0.8"))
    vosk_model_path: str = os.getenv(
        "VOSK_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "vosk-model-small-en-us-0.15"),
    )
    vosk_keyword: str = os.getenv("VOSK_KEYWORD", "grace")
    vosk_threshold: float = float(os.getenv("VOSK_THRESHOLD", "0.4"))

    # Whisper
    whisper_model_path: str = os.getenv("WHISPER_MODEL_PATH", "small")
    whisper_vad_threshold: float = float(os.getenv("WHISPER_VAD_THRESHOLD", "0.008"))
    whisper_silence_duration_ms: int = int(os.getenv("WHISPER_SILENCE_DURATION_MS", "700"))

    # Kokoro
    kokoro_workers: int = int(os.getenv("KOKORO_WORKERS", "3"))
    kokoro_device: str = os.getenv("KOKORO_DEVICE", "cuda")
    kokoro_dtype: str = os.getenv("KOKORO_DTYPE", "float16")

    # WebSocket (frontend)
    ws_host: str = os.getenv("WS_HOST", "127.0.0.1")
    ws_port: int = int(os.getenv("WS_PORT", "8765"))

    # Derived
    @property
    def llama_server_url(self) -> str:
        return f"http://{self.llama_host}:{self.llama_port}"
