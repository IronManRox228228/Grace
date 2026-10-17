"""Per-slot model configuration and fleet presets for Grace (PLAN §3, §5, Task 13).

Provides:
- SlotConfig: per-slot model name, runtime, thresholds, and margin.
- FleetConfig: collection of all slots for a given preset ("gpu-single" vs "cpu-fleet").
- Presets:
  * "gpu-single": Buildathon single-model setup (Holo3.1-4B serving all slots).
  * "cpu-fleet": Micromodel fleet on CPU floor:
      - Tool calling: LFM2.5-350M / FunctionGemma 270M (GGUF / llama_cpp)
      - Element selection: GLiNER2.5-Decide / Julia-1 (ONNX Int8)
      - Yes/No: GLiNER2.5-Decide / Contrastive Head (ONNX)
      - Routing: Contrastive Head / Fast-path
      - Vision: LFM2.5-VL-450M (GGUF)
      - Midsize fallback: LFM2.5-1.2B-Instruct (GGUF, loaded on-demand only)
- Tier-driven default selection via detect_hardware_tier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from grace.tier import HardwareTier, detect_hardware_tier


@dataclass(frozen=True)
class SlotConfig:
    """Configuration for a specific micromodel slot."""
    slot_name: str
    model_name: str
    model_path: str
    runtime: str  # "llama_cpp", "onnx", "ort", "transformers", "internal"
    threshold: float = 0.70
    margin: float = 0.15
    context_window: int = 2048
    temperature: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FleetConfig:
    """Configured model fleet for Grace."""
    preset: str
    slots: Dict[str, SlotConfig]

    def get_slot(self, slot_name: str) -> SlotConfig:
        if slot_name not in self.slots:
            raise KeyError(f"Unknown slot '{slot_name}'. Available: {list(self.slots.keys())}")
        return self.slots[slot_name]

    @property
    def is_fleet(self) -> bool:
        return self.preset == "cpu-fleet"

    @property
    def is_single(self) -> bool:
        return self.preset == "gpu-single"


def build_gpu_single_preset(
    holo_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
) -> FleetConfig:
    """Buildathon single-model preset: Holo3.1-4B runs all slots."""
    default_holo = holo_path or os.getenv(
        "LLAMA_MODEL_PATH",
        os.path.join(os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.Q4_K_M.gguf"),
    )
    default_mmproj = mmproj_path or os.getenv(
        "LLAMA_MMPROJ_PATH",
        os.path.join(os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.mmproj-q8_0.gguf"),
    )

    common_slot = lambda name: SlotConfig(
        slot_name=name,
        model_name="Holo-3.1-4B",
        model_path=default_holo,
        runtime="llama_cpp",
        threshold=0.50,
        margin=0.10,
        context_window=8192,
        extra={"mmproj": default_mmproj},
    )

    slots = {
        "tool_calling": common_slot("tool_calling"),
        "element_selection": common_slot("element_selection"),
        "yes_no": common_slot("yes_no"),
        "routing": common_slot("routing"),
        "vision": common_slot("vision"),
        "midsize_fallback": common_slot("midsize_fallback"),
    }
    return FleetConfig(preset="gpu-single", slots=slots)


def build_cpu_fleet_preset(models_dir: Optional[str] = None) -> FleetConfig:
    """Micromodel fleet preset for the CPU floor and edge environments."""
    base_dir = models_dir or os.path.join(os.path.dirname(__file__), "..", "..", "models")

    slots = {
        "tool_calling": SlotConfig(
            slot_name="tool_calling",
            model_name="LFM2.5-350M-ToolCaller",
            model_path=os.path.join(base_dir, "fleet", "lfm-350m-tool-caller.gguf"),
            runtime="llama_cpp",
            threshold=0.75,
            margin=0.15,
            context_window=2048,
            temperature=0.0,
            extra={"challenger": "FunctionGemma-270M"},
        ),
        "element_selection": SlotConfig(
            slot_name="element_selection",
            model_name="GLiNER2.5-Decide-340M",
            model_path=os.path.join(base_dir, "fleet", "gliner-decide-340m.onnx"),
            runtime="onnx",
            threshold=0.70,
            margin=0.15,
            context_window=1024,
            extra={"challenger": "Julia-1-ONNX", "max_candidates": 20},
        ),
        "yes_no": SlotConfig(
            slot_name="yes_no",
            model_name="GLiNER2.5-Decide-YesNo",
            model_path=os.path.join(base_dir, "fleet", "gliner-decide-340m.onnx"),
            runtime="onnx",
            threshold=0.80,
            margin=0.20,
            context_window=512,
        ),
        "routing": SlotConfig(
            slot_name="routing",
            model_name="Contrastive-Routing-Head",
            model_path=os.path.join(base_dir, "fleet", "contrastive_head.onnx"),
            runtime="onnx",
            threshold=0.75,
            margin=0.15,
            context_window=512,
        ),
        "vision": SlotConfig(
            slot_name="vision",
            model_name="LFM2.5-VL-450M",
            model_path=os.path.join(base_dir, "fleet", "lfm-vl-450m.gguf"),
            runtime="llama_cpp",
            threshold=0.65,
            margin=0.10,
            context_window=2048,
            extra={"challenger": "Intern-Decision-0.8B"},
        ),
        "midsize_fallback": SlotConfig(
            slot_name="midsize_fallback",
            model_name="LFM2.5-1.2B-Instruct",
            model_path=os.path.join(base_dir, "fleet", "lfm-1.2b-instruct.gguf"),
            runtime="llama_cpp",
            threshold=0.70,
            margin=0.10,
            context_window=4096,
            extra={"on_demand_only": True},
        ),
    }
    return FleetConfig(preset="cpu-fleet", slots=slots)


def get_fleet_config(
    preset_name: Optional[str] = None,
    config: Optional[Any] = None,
) -> FleetConfig:
    """Retrieve FleetConfig based on preset override or hardware tier detection."""
    name = preset_name
    if not name and config is not None:
        if getattr(config, "use_cloud_llm", False):
            return build_gpu_single_preset()
        name = getattr(config, "preset", None)
    if not name:
        name = os.getenv("GRACE_PRESET")

    if name:
        name_clean = name.strip().lower()
        if name_clean in ("gpu-single", "single", "buildathon"):
            return build_gpu_single_preset()
        elif name_clean in ("cpu-fleet", "fleet"):
            return build_cpu_fleet_preset()

    # Tier-driven auto selection:
    tier_diag = detect_hardware_tier(config)
    if tier_diag.tier == HardwareTier.FLOOR_CPU or tier_diag.tier == HardwareTier.MIDDLE_IGPU:
        # Default to CPU fleet for resource efficiency
        return build_cpu_fleet_preset()
    else:
        # On high NPU tier, GPU single preset can be used if configured
        return build_gpu_single_preset()
