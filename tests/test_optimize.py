from guitartab_ai.config import FretboardConfig, OptimizerConfig
from guitartab_ai.fretboard import GuitarFretboard
from guitartab_ai.models import NoteEvent
from guitartab_ai.optimize import FingeringOptimizer


def _note(start_frame: int, midi: int) -> NoteEvent:
    start_s = start_frame * 0.1
    end_s = start_s + 0.1
    return NoteEvent(
        start_frame=start_frame,
        end_frame=start_frame + 1,
        start_s=start_s,
        end_s=end_s,
        midi=midi,
        note_name="",
        mean_confidence=0.95,
        mean_frequency_hz=0.0,
        frame_count=1,
    )


def test_optimizer_prefers_consistent_position_for_stepwise_phrase() -> None:
    notes = [_note(0, 64), _note(1, 65), _note(2, 67)]
    fretboard = GuitarFretboard.from_config(FretboardConfig(min_fret=0, max_fret=20))
    optimizer = FingeringOptimizer(fretboard=fretboard, config=OptimizerConfig())

    result = optimizer.optimize(notes)

    assert [(decision.position.string_number, decision.position.fret) for decision in result.decisions] == [
        (1, 0),
        (1, 1),
        (1, 3),
    ]


def test_optimizer_keeps_repeated_note_on_same_position() -> None:
    notes = [_note(0, 64), _note(1, 64), _note(2, 64)]
    fretboard = GuitarFretboard.from_config(FretboardConfig(min_fret=0, max_fret=20))
    optimizer = FingeringOptimizer(fretboard=fretboard, config=OptimizerConfig())

    result = optimizer.optimize(notes)

    assert len({(decision.position.string_number, decision.position.fret) for decision in result.decisions}) == 1
