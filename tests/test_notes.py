from guitartab_ai.config import NoteExtractionConfig
from guitartab_ai.models import PitchFrame
from guitartab_ai.music import midi_to_hz
from guitartab_ai.notes import extract_note_events


def _frame(index: int, midi: int | None, *, confidence: float = 0.9, hop_s: float = 0.01) -> PitchFrame:
    frequency = midi_to_hz(midi) if midi is not None else None
    return PitchFrame(
        index=index,
        time_s=index * hop_s,
        frequency_hz=frequency,
        confidence=confidence,
        midi_continuous=float(midi) if midi is not None else None,
        midi_rounded=midi,
        voiced=midi is not None,
    )


def test_extract_note_events_bridges_short_confidence_gap() -> None:
    frames = [_frame(index, 64) for index in range(5)]
    frames.append(_frame(5, None, confidence=0.05))
    frames.extend(_frame(index, 64) for index in range(6, 11))

    notes = extract_note_events(frames, NoteExtractionConfig(max_gap_ms=30.0, merge_same_pitch_gap_ms=30.0))

    assert len(notes) == 1
    assert notes[0].midi == 64
    assert notes[0].frame_count == 11


def test_extract_note_events_suppresses_brief_pitch_jitter() -> None:
    frames = [_frame(index, 64) for index in range(5)]
    frames.extend(_frame(index, 65) for index in range(5, 7))
    frames.extend(_frame(index, 64) for index in range(7, 12))

    notes = extract_note_events(
        frames,
        NoteExtractionConfig(pitch_stability_frames=3, min_note_ms=20.0),
    )

    assert len(notes) == 1
    assert notes[0].midi == 64


def test_extract_note_events_drops_isolated_short_note() -> None:
    frames = [_frame(index, 64) for index in range(5)]
    frames.extend(_frame(index, 67) for index in range(5, 7))
    frames.extend(_frame(index, 69) for index in range(7, 13))

    notes = extract_note_events(
        frames,
        NoteExtractionConfig(pitch_stability_frames=1, min_note_ms=40.0),
    )

    assert [note.midi for note in notes] == [64, 69]
