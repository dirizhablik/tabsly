from guitartab_ai.music import hz_to_midi, midi_to_hz, midi_to_note_name, quantize_midi


def test_hz_to_midi_and_back_round_trips_for_a4() -> None:
    midi = hz_to_midi(440.0)
    assert round(midi, 6) == 69.0
    assert round(midi_to_hz(midi), 6) == 440.0


def test_quantize_midi_rounds_to_nearest_semitone() -> None:
    assert quantize_midi(63.51) == 64
    assert quantize_midi(63.49) == 63


def test_midi_to_note_name_formats_octave() -> None:
    assert midi_to_note_name(40) == "E2"
    assert midi_to_note_name(64) == "E4"
