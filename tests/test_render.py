from guitartab_ai.config import RenderConfig
from guitartab_ai.models import FingeringDecision, GuitarPosition, NoteEvent
from guitartab_ai.render import render_ascii_tab


def _decision(string_number: int, fret: int, midi: int, start_s: float) -> FingeringDecision:
    note = NoteEvent(
        start_frame=0,
        end_frame=1,
        start_s=start_s,
        end_s=start_s + 0.1,
        midi=midi,
        note_name="E4",
        mean_confidence=0.9,
        mean_frequency_hz=329.63,
        frame_count=1,
    )
    position = GuitarPosition(
        string_number=string_number,
        string_name="E4",
        open_midi=64,
        fret=fret,
        midi=midi,
        note_name="E4",
    )
    return FingeringDecision(note=note, position=position, intrinsic_cost=0.0, transition_cost=0.0, cumulative_cost=0.0)


def test_render_ascii_tab_places_frets_on_expected_strings() -> None:
    tab = render_ascii_tab(
        [
            _decision(1, 0, 64, 0.0),
            _decision(1, 2, 66, 0.1),
            _decision(2, 5, 64, 0.2),
        ],
        RenderConfig(include_note_listing=False),
    )

    lines = tab.splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("e|0--2--")
    assert lines[1].endswith("5--")
    assert all(line.count("|") == 1 for line in lines)
