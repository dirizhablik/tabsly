from guitartab_ai.config import FretboardConfig
from guitartab_ai.fretboard import GuitarFretboard


def test_positions_for_midi_returns_all_standard_candidates() -> None:
    fretboard = GuitarFretboard.from_config(FretboardConfig(min_fret=0, max_fret=20))

    positions = fretboard.positions_for_midi(64)
    rendered = {(position.string_number, position.fret) for position in positions}

    assert rendered == {(1, 0), (2, 5), (3, 9), (4, 14), (5, 19)}


def test_positions_for_midi_returns_empty_when_note_outside_fret_range() -> None:
    fretboard = GuitarFretboard.from_config(FretboardConfig(min_fret=0, max_fret=5))

    assert fretboard.positions_for_midi(90) == []
