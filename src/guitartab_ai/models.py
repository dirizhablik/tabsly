from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FloatArray = np.ndarray


@dataclass(slots=True, frozen=True)
class AudioBuffer:
    """Normalized mono audio loaded for inference."""

    samples: FloatArray
    sample_rate: int
    duration_s: float


@dataclass(slots=True, frozen=True)
class PitchFrame:
    """Single frame of pitch tracking output."""

    index: int
    time_s: float
    frequency_hz: float | None
    confidence: float
    midi_continuous: float | None
    midi_rounded: int | None
    voiced: bool


@dataclass(slots=True, frozen=True)
class NoteEvent:
    """Monophonic note event derived from frame-level pitch."""

    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    midi: int
    note_name: str
    mean_confidence: float
    mean_frequency_hz: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(slots=True, frozen=True)
class GuitarString:
    """Physical guitar string in standard notation."""

    string_number: int
    name: str
    open_midi: int


@dataclass(slots=True, frozen=True)
class GuitarPosition:
    """A single playable fretboard location for a note."""

    string_number: int
    string_name: str
    open_midi: int
    fret: int
    midi: int
    note_name: str

    @property
    def line_index(self) -> int:
        """Return the 0-based ASCII tab line index, top to bottom."""

        return 6 - self.string_number

    @property
    def effective_fret(self) -> int:
        """Anchor used for left-hand position modeling."""

        return self.fret if self.fret > 0 else 1


@dataclass(slots=True, frozen=True)
class FingeringDecision:
    """Chosen fretboard position for a note event."""

    note: NoteEvent
    position: GuitarPosition
    intrinsic_cost: float
    transition_cost: float
    cumulative_cost: float


@dataclass(slots=True, frozen=True)
class OptimizationResult:
    """Globally optimized fingering path."""

    decisions: tuple[FingeringDecision, ...]
    total_cost: float


@dataclass(slots=True, frozen=True)
class TranscriptionResult:
    """Final result returned by the transcription pipeline."""

    audio_path: Path
    sample_rate: int
    duration_s: float
    pitch_frames: tuple[PitchFrame, ...]
    note_events: tuple[NoteEvent, ...]
    optimization: OptimizationResult
    ascii_tab: str
