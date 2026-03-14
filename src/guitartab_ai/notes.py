from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import NoteExtractionConfig
from .models import NoteEvent, PitchFrame
from .music import midi_to_note_name, quantize_midi


def extract_note_events(frames: Sequence[PitchFrame], config: NoteExtractionConfig) -> list[NoteEvent]:
    """Convert noisy frame-level pitch into stable note events."""

    if not frames:
        return []

    hop_s = _frame_hop_s(frames)
    raw_labels = _quantized_labels(frames, config.midi_smoothing_window)
    gap_frames = _ms_to_frames(config.max_gap_ms, hop_s)
    labels = _bridge_short_gaps(raw_labels, gap_frames)
    labels = _suppress_short_note_runs(labels, config.pitch_stability_frames)
    labels = _bridge_short_gaps(labels, gap_frames)

    events = _labels_to_events(labels, frames, hop_s)
    min_note_s = config.min_note_ms / 1_000.0
    merge_gap_s = config.merge_same_pitch_gap_ms / 1_000.0
    return _cleanup_note_events(events, min_note_s=min_note_s, merge_gap_s=merge_gap_s)


def _quantized_labels(frames: Sequence[PitchFrame], smoothing_window: int) -> list[int | None]:
    smoothed_midi = _smooth_voiced_midis(frames, smoothing_window)
    labels: list[int | None] = []
    for frame, midi_value in zip(frames, smoothed_midi, strict=True):
        if not frame.voiced or midi_value is None:
            labels.append(None)
            continue
        labels.append(quantize_midi(midi_value))
    return labels


def _smooth_voiced_midis(frames: Sequence[PitchFrame], smoothing_window: int) -> list[float | None]:
    if smoothing_window <= 1:
        return [frame.midi_continuous if frame.voiced else None for frame in frames]

    result: list[float | None] = [None] * len(frames)
    start = 0
    while start < len(frames):
        if not frames[start].voiced or frames[start].midi_continuous is None:
            start += 1
            continue
        end = start + 1
        while end < len(frames) and frames[end].voiced and frames[end].midi_continuous is not None:
            end += 1
        segment = [float(frames[index].midi_continuous) for index in range(start, end)]
        smoothed = _median_smooth_segment(segment, smoothing_window)
        for offset, value in enumerate(smoothed):
            result[start + offset] = value
        start = end
    return result


def _median_smooth_segment(values: Sequence[float], window: int) -> list[float]:
    radius = max(0, window // 2)
    smoothed: list[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        smoothed.append(float(np.median(values[left:right])))
    return smoothed


def _bridge_short_gaps(labels: Sequence[int | None], max_gap_frames: int) -> list[int | None]:
    if max_gap_frames <= 0:
        return list(labels)

    bridged = list(labels)
    index = 0
    while index < len(bridged):
        if bridged[index] is not None:
            index += 1
            continue
        gap_start = index
        while index < len(bridged) and bridged[index] is None:
            index += 1
        gap_end = index
        gap_length = gap_end - gap_start
        prev_label = bridged[gap_start - 1] if gap_start > 0 else None
        next_label = bridged[gap_end] if gap_end < len(bridged) else None
        if 0 < gap_length <= max_gap_frames and prev_label is not None and prev_label == next_label:
            for fill_index in range(gap_start, gap_end):
                bridged[fill_index] = prev_label
    return bridged


def _suppress_short_note_runs(labels: Sequence[int | None], min_run_frames: int) -> list[int | None]:
    if min_run_frames <= 1:
        return list(labels)

    cleaned = list(labels)
    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(cleaned):
            current = cleaned[index]
            run_end = index + 1
            while run_end < len(cleaned) and cleaned[run_end] == current:
                run_end += 1

            if current is not None and (run_end - index) < min_run_frames:
                prev_label = cleaned[index - 1] if index > 0 else None
                next_label = cleaned[run_end] if run_end < len(cleaned) else None
                replacement = _choose_replacement(current, prev_label, next_label)
                for replace_index in range(index, run_end):
                    cleaned[replace_index] = replacement
                changed = True
            index = run_end
    return cleaned


def _choose_replacement(current: int, prev_label: int | None, next_label: int | None) -> int | None:
    if prev_label is not None and prev_label == next_label:
        return prev_label
    if prev_label is None:
        return next_label
    if next_label is None:
        return prev_label
    return prev_label if abs(prev_label - current) <= abs(next_label - current) else next_label


def _labels_to_events(
    labels: Sequence[int | None],
    frames: Sequence[PitchFrame],
    hop_s: float,
) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    index = 0
    while index < len(labels):
        label = labels[index]
        if label is None:
            index += 1
            continue
        end = index + 1
        while end < len(labels) and labels[end] == label:
            end += 1

        segment_frames = frames[index:end]
        voiced_frequencies = [frame.frequency_hz for frame in segment_frames if frame.frequency_hz is not None]
        mean_frequency_hz = float(np.mean(voiced_frequencies)) if voiced_frequencies else 0.0
        mean_confidence = float(np.mean([frame.confidence for frame in segment_frames]))
        start_s = frames[index].time_s
        end_s = frames[end - 1].time_s + hop_s
        events.append(
            NoteEvent(
                start_frame=index,
                end_frame=end,
                start_s=start_s,
                end_s=end_s,
                midi=label,
                note_name=midi_to_note_name(label),
                mean_confidence=mean_confidence,
                mean_frequency_hz=mean_frequency_hz,
                frame_count=end - index,
            )
        )
        index = end
    return events


def _cleanup_note_events(
    events: Sequence[NoteEvent],
    *,
    min_note_s: float,
    merge_gap_s: float,
) -> list[NoteEvent]:
    if not events:
        return []

    working = list(events)
    cleaned: list[NoteEvent] = []
    index = 0
    while index < len(working):
        event = working[index]
        next_event = working[index + 1] if index + 1 < len(working) else None
        previous_event = cleaned[-1] if cleaned else None

        if event.duration_s < min_note_s:
            if previous_event and next_event and previous_event.midi == next_event.midi:
                if event.start_s - previous_event.end_s <= merge_gap_s and next_event.start_s - event.end_s <= merge_gap_s:
                    cleaned[-1] = _merge_events(previous_event, next_event)
                    index += 2
                    continue
            if previous_event and previous_event.midi == event.midi and event.start_s - previous_event.end_s <= merge_gap_s:
                cleaned[-1] = _merge_events(previous_event, event)
                index += 1
                continue
            if next_event and next_event.midi == event.midi and next_event.start_s - event.end_s <= merge_gap_s:
                working[index + 1] = _merge_events(event, next_event)
                index += 1
                continue
            index += 1
            continue

        if previous_event and previous_event.midi == event.midi and event.start_s - previous_event.end_s <= merge_gap_s:
            cleaned[-1] = _merge_events(previous_event, event)
        else:
            cleaned.append(event)
        index += 1
    return cleaned


def _merge_events(left: NoteEvent, right: NoteEvent) -> NoteEvent:
    total_frames = left.frame_count + right.frame_count
    confidence = ((left.mean_confidence * left.frame_count) + (right.mean_confidence * right.frame_count)) / total_frames
    frequency_hz = ((left.mean_frequency_hz * left.frame_count) + (right.mean_frequency_hz * right.frame_count)) / total_frames
    return NoteEvent(
        start_frame=left.start_frame,
        end_frame=right.end_frame,
        start_s=left.start_s,
        end_s=right.end_s,
        midi=left.midi,
        note_name=left.note_name,
        mean_confidence=float(confidence),
        mean_frequency_hz=float(frequency_hz),
        frame_count=total_frames,
    )


def _frame_hop_s(frames: Sequence[PitchFrame]) -> float:
    if len(frames) < 2:
        return 0.01
    return max(frames[1].time_s - frames[0].time_s, 1e-6)


def _ms_to_frames(milliseconds: float, hop_s: float) -> int:
    if milliseconds <= 0:
        return 0
    return max(1, int(round((milliseconds / 1_000.0) / hop_s)))
