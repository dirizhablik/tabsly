from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import FretboardConfig
from .models import GuitarPosition, GuitarString, NoteEvent
from .music import midi_to_note_name


STANDARD_STRING_NUMBERS = (6, 5, 4, 3, 2, 1)


@dataclass(slots=True)
class GuitarFretboard:
    """Standard-tuned six-string fretboard."""

    strings: tuple[GuitarString, ...]
    min_fret: int
    max_fret: int

    @classmethod
    def from_config(cls, config: FretboardConfig) -> "GuitarFretboard":
        strings = tuple(
            GuitarString(
                string_number=string_number,
                name=name,
                open_midi=open_midi,
            )
            for string_number, name, open_midi in zip(
                STANDARD_STRING_NUMBERS,
                config.string_names,
                config.tuning,
                strict=True,
            )
        )
        return cls(strings=strings, min_fret=config.min_fret, max_fret=config.max_fret)

    def positions_for_midi(self, midi: int) -> list[GuitarPosition]:
        """Return every playable string/fret location for a note."""

        positions: list[GuitarPosition] = []
        for guitar_string in self.strings:
            fret = midi - guitar_string.open_midi
            if self.min_fret <= fret <= self.max_fret:
                positions.append(
                    GuitarPosition(
                        string_number=guitar_string.string_number,
                        string_name=guitar_string.name,
                        open_midi=guitar_string.open_midi,
                        fret=fret,
                        midi=midi,
                        note_name=midi_to_note_name(midi),
                    )
                )
        return sorted(positions, key=lambda position: (position.fret, -position.string_number))

    def candidate_matrix(self, notes: Sequence[NoteEvent]) -> list[list[GuitarPosition]]:
        matrix = [self.positions_for_midi(note.midi) for note in notes]
        for note, positions in zip(notes, matrix, strict=True):
            if not positions:
                raise ValueError(
                    f"No playable fretboard positions for {note.note_name} (MIDI {note.midi}) "
                    f"within frets {self.min_fret}-{self.max_fret}."
                )
        return matrix
