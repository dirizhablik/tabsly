from __future__ import annotations

import math

NOTE_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
A4_MIDI = 69
A4_HZ = 440.0


def hz_to_midi(frequency_hz: float) -> float:
    """Convert frequency in Hz to fractional MIDI number."""

    if frequency_hz <= 0:
        raise ValueError("Frequency must be positive.")
    return A4_MIDI + 12.0 * math.log2(frequency_hz / A4_HZ)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI number to frequency in Hz."""

    return A4_HZ * (2.0 ** ((midi - A4_MIDI) / 12.0))


def quantize_midi(midi_value: float) -> int:
    """Round a fractional MIDI estimate to the nearest semitone."""

    return int(round(midi_value))


def midi_to_note_name(midi: int) -> str:
    """Format MIDI pitch as note name plus octave."""

    octave = (midi // 12) - 1
    return f"{NOTE_NAMES_SHARP[midi % 12]}{octave}"
