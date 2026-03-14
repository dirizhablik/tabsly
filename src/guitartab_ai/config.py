from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import tomllib


@dataclass(slots=True)
class AudioConfig:
    target_sample_rate: int = 16_000
    mono: bool = True
    normalize_peak: bool = True
    resample_type: str = "soxr_hq"


@dataclass(slots=True)
class PitchConfig:
    backend: str = "torchcrepe"
    frame_hop_ms: float = 10.0
    fmin_hz: float = 70.0
    fmax_hz: float = 1_400.0
    confidence_threshold: float = 0.35
    model_capacity: str = "full"
    batch_size: int = 2_048
    device: str = "auto"
    median_filter_width: int = 5


@dataclass(slots=True)
class NoteExtractionConfig:
    min_note_ms: float = 80.0
    max_gap_ms: float = 30.0
    merge_same_pitch_gap_ms: float = 45.0
    pitch_stability_frames: int = 3
    midi_smoothing_window: int = 5


@dataclass(slots=True)
class FretboardConfig:
    min_fret: int = 0
    max_fret: int = 20
    tuning: tuple[int, int, int, int, int, int] = (40, 45, 50, 55, 59, 64)
    string_names: tuple[str, str, str, str, str, str] = ("E2", "A2", "D3", "G3", "B3", "E4")


@dataclass(slots=True)
class OptimizerConfig:
    position_cost_weight: float = 0.08
    high_fret_penalty_weight: float = 0.12
    open_string_bonus: float = -0.15
    fret_distance_weight: float = 1.0
    string_change_weight: float = 0.35
    same_string_motion_weight: float = 0.15
    large_shift_threshold: int = 4
    large_shift_penalty_weight: float = 0.8
    repeated_note_position_change_penalty: float = 1.25
    string_cross_shift_weight: float = 0.12
    open_string_shift_relief: float = 0.75
    max_time_relief_s: float = 0.35
    time_relief_discount: float = 0.25


@dataclass(slots=True)
class RenderConfig:
    min_cell_width: int = 2
    note_spacing: int = 1
    include_note_listing: bool = False


@dataclass(slots=True)
class TranscriptionConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    notes: NoteExtractionConfig = field(default_factory=NoteExtractionConfig)
    fretboard: FretboardConfig = field(default_factory=FretboardConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    debug: bool = False

    @classmethod
    def from_toml(cls, path: str | Path) -> "TranscriptionConfig":
        """Load nested configuration from a TOML file."""

        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        config = cls()
        section_map = {
            "audio": AudioConfig,
            "pitch": PitchConfig,
            "notes": NoteExtractionConfig,
            "fretboard": FretboardConfig,
            "optimizer": OptimizerConfig,
            "render": RenderConfig,
        }
        for section_name in section_map:
            section_data = raw.get(section_name)
            if not section_data:
                continue
            current_section = getattr(config, section_name)
            setattr(config, section_name, _replace_dataclass(current_section, section_data))
        if "debug" in raw:
            config.debug = bool(raw["debug"])
        return config


def _replace_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    field_map = instance.__dataclass_fields__
    cleaned: dict[str, Any] = {}
    for key, value in updates.items():
        if key not in field_map:
            continue
        current_value = getattr(instance, key)
        if isinstance(current_value, tuple) and isinstance(value, list):
            cleaned[key] = tuple(value)
        else:
            cleaned[key] = value
    return replace(instance, **cleaned)
