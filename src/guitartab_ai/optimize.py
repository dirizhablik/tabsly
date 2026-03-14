from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from .config import OptimizerConfig
from .fretboard import GuitarFretboard
from .models import FingeringDecision, GuitarPosition, NoteEvent, OptimizationResult


@dataclass(slots=True)
class FingeringOptimizer:
    """Dynamic-programming search over candidate fretboard positions."""

    fretboard: GuitarFretboard
    config: OptimizerConfig

    def optimize(self, notes: Sequence[NoteEvent]) -> OptimizationResult:
        if not notes:
            return OptimizationResult(decisions=tuple(), total_cost=0.0)

        candidates = self.fretboard.candidate_matrix(notes)
        dp = [[inf for _ in note_candidates] for note_candidates in candidates]
        backpointer: list[list[int | None]] = [[None for _ in note_candidates] for note_candidates in candidates]

        for candidate_index, position in enumerate(candidates[0]):
            dp[0][candidate_index] = self._intrinsic_cost(position)

        for note_index in range(1, len(notes)):
            current_note = notes[note_index]
            previous_note = notes[note_index - 1]
            for current_index, current_position in enumerate(candidates[note_index]):
                intrinsic = self._intrinsic_cost(current_position)
                best_cost = inf
                best_prev_index: int | None = None
                for previous_index, previous_position in enumerate(candidates[note_index - 1]):
                    transition = self._transition_cost(
                        previous_position,
                        current_position,
                        previous_note=previous_note,
                        current_note=current_note,
                    )
                    total_cost = dp[note_index - 1][previous_index] + intrinsic + transition
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_prev_index = previous_index
                dp[note_index][current_index] = best_cost
                backpointer[note_index][current_index] = best_prev_index

        final_candidates = dp[-1]
        final_index = min(range(len(final_candidates)), key=final_candidates.__getitem__)
        total_cost = final_candidates[final_index]

        chosen_indices = [0] * len(notes)
        chosen_indices[-1] = final_index
        for note_index in range(len(notes) - 1, 0, -1):
            previous_index = backpointer[note_index][chosen_indices[note_index]]
            if previous_index is None:
                raise RuntimeError("Optimization backpointer is incomplete.")
            chosen_indices[note_index - 1] = previous_index

        decisions: list[FingeringDecision] = []
        cumulative_cost = 0.0
        previous_position: GuitarPosition | None = None
        previous_note: NoteEvent | None = None
        for note, note_candidates, chosen_index in zip(notes, candidates, chosen_indices, strict=True):
            position = note_candidates[chosen_index]
            intrinsic = self._intrinsic_cost(position)
            transition = (
                0.0
                if previous_position is None or previous_note is None
                else self._transition_cost(previous_position, position, previous_note=previous_note, current_note=note)
            )
            cumulative_cost += intrinsic + transition
            decisions.append(
                FingeringDecision(
                    note=note,
                    position=position,
                    intrinsic_cost=intrinsic,
                    transition_cost=transition,
                    cumulative_cost=cumulative_cost,
                )
            )
            previous_position = position
            previous_note = note

        return OptimizationResult(decisions=tuple(decisions), total_cost=total_cost)

    def _intrinsic_cost(self, position: GuitarPosition) -> float:
        cost = self.config.position_cost_weight * float(position.fret)
        if position.fret > 12:
            cost += self.config.high_fret_penalty_weight * float(position.fret - 12)
        if position.fret == 0:
            cost += self.config.open_string_bonus
        return cost

    def _transition_cost(
        self,
        previous: GuitarPosition,
        current: GuitarPosition,
        *,
        previous_note: NoteEvent,
        current_note: NoteEvent,
    ) -> float:
        fret_shift = abs(current.effective_fret - previous.effective_fret)
        string_shift = abs(current.string_number - previous.string_number)

        cost = 0.0
        cost += fret_shift * self.config.fret_distance_weight
        cost += string_shift * self.config.string_change_weight

        if previous.string_number == current.string_number:
            cost += abs(current.fret - previous.fret) * self.config.same_string_motion_weight

        if fret_shift > self.config.large_shift_threshold:
            overflow = fret_shift - self.config.large_shift_threshold
            cost += (overflow**2) * self.config.large_shift_penalty_weight

        if string_shift > 0 and fret_shift > 0:
            cost += string_shift * fret_shift * self.config.string_cross_shift_weight

        if previous.midi == current.midi and previous != current:
            cost += self.config.repeated_note_position_change_penalty

        if previous.fret == 0 or current.fret == 0:
            cost *= self.config.open_string_shift_relief

        move_window_s = previous_note.duration_s + max(0.0, current_note.start_s - previous_note.end_s)
        if move_window_s > 0.0:
            relief_ratio = min(move_window_s / self.config.max_time_relief_s, 1.0)
            cost *= 1.0 - (relief_ratio * self.config.time_relief_discount)

        return cost
