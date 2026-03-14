from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio import load_audio
from .config import TranscriptionConfig
from .fretboard import GuitarFretboard
from .models import TranscriptionResult
from .notes import extract_note_events
from .optimize import FingeringOptimizer
from .pitch import PitchEstimator, build_pitch_estimator
from .render import render_ascii_tab


@dataclass(slots=True)
class TranscriptionPipeline:
    """End-to-end orchestration for clean monophonic guitar transcription."""

    config: TranscriptionConfig
    pitch_estimator: PitchEstimator | None = None

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        path = Path(audio_path)
        audio = load_audio(path, self.config.audio)
        estimator = self.pitch_estimator or build_pitch_estimator(self.config.pitch)
        pitch_frames = estimator.estimate(audio, self.config.pitch)
        note_events = extract_note_events(pitch_frames, self.config.notes)
        if not note_events:
            raise ValueError(
                "No note events survived extraction. Lower the confidence threshold or provide cleaner audio."
            )

        fretboard = GuitarFretboard.from_config(self.config.fretboard)
        optimization = FingeringOptimizer(fretboard=fretboard, config=self.config.optimizer).optimize(note_events)
        ascii_tab = render_ascii_tab(optimization.decisions, self.config.render)

        return TranscriptionResult(
            audio_path=path,
            sample_rate=audio.sample_rate,
            duration_s=audio.duration_s,
            pitch_frames=tuple(pitch_frames),
            note_events=tuple(note_events),
            optimization=optimization,
            ascii_tab=ascii_tab,
        )
