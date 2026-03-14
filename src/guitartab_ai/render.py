from __future__ import annotations

from typing import Sequence

from .config import RenderConfig
from .models import FingeringDecision


TAB_ORDER = (
    (1, "e"),
    (2, "B"),
    (3, "G"),
    (4, "D"),
    (5, "A"),
    (6, "E"),
)


def render_ascii_tab(decisions: Sequence[FingeringDecision], config: RenderConfig) -> str:
    """Render a playable monophonic path as six-line ASCII tablature."""

    lines = {string_number: [f"{name}|"] for string_number, name in TAB_ORDER}
    note_listing: list[str] = []

    for decision in decisions:
        fret_text = str(decision.position.fret)
        cell_width = max(len(fret_text), config.min_cell_width) + config.note_spacing
        for string_number, _ in TAB_ORDER:
            if string_number == decision.position.string_number:
                padding = "-" * (cell_width - len(fret_text))
                lines[string_number].append(f"{fret_text}{padding}")
            else:
                lines[string_number].append("-" * cell_width)

        if config.include_note_listing:
            note_listing.append(
                f"{decision.note.start_s:6.3f}-{decision.note.end_s:6.3f}s "
                f"{decision.note.note_name:<4} -> string {decision.position.string_number} fret {decision.position.fret}"
            )

    rendered_lines = ["".join(lines[string_number]) for string_number, _ in TAB_ORDER]
    if not config.include_note_listing or not note_listing:
        return "\n".join(rendered_lines)

    listing_header = "\n".join(["", "Notes:", *note_listing])
    return "\n".join(rendered_lines) + listing_header
