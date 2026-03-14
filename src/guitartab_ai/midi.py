from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .models import FingeringDecision


def write_midi(decisions: Sequence[FingeringDecision], path: str | Path) -> None:
    """Write note events to a MIDI file for quick audible inspection."""

    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError("pretty_midi is required for MIDI export.") from exc

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=27, name="Guitar Tab AI")
    for decision in decisions:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=96,
                pitch=decision.note.midi,
                start=decision.note.start_s,
                end=decision.note.end_s,
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(Path(path)))
