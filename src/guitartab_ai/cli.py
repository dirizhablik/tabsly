from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import TranscriptionConfig
from .midi import write_midi
from .pipeline import TranscriptionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guitartab-ai",
        description="Transcribe clean monophonic guitar audio into playable ASCII tablature.",
    )
    parser.add_argument("input", help="Path to input audio file.")
    parser.add_argument("--out", help="Path to write ASCII tab. Defaults to stdout.")
    parser.add_argument("--config", help="Optional TOML config file.")
    parser.add_argument("--min-fret", type=int, help="Lowest fret considered playable.")
    parser.add_argument("--max-fret", type=int, help="Highest fret considered playable.")
    parser.add_argument("--confidence-threshold", type=float, help="Minimum pitch confidence to treat a frame as voiced.")
    parser.add_argument("--min-note-ms", type=float, help="Minimum note duration after cleanup.")
    parser.add_argument("--device", help="Pitch inference device: auto, cpu, or cuda.")
    parser.add_argument("--debug", action="store_true", help="Print note-to-position decisions to stderr.")
    parser.add_argument("--emit-midi", help="Optional MIDI output path for sanity checking.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TranscriptionConfig.from_toml(args.config) if args.config else TranscriptionConfig()
    config = apply_cli_overrides(config, args)

    pipeline = TranscriptionPipeline(config=config)
    result = pipeline.transcribe(args.input)

    if args.out:
        Path(args.out).write_text(result.ascii_tab + "\n", encoding="utf-8")
    else:
        print(result.ascii_tab)

    if args.emit_midi:
        write_midi(result.optimization.decisions, args.emit_midi)

    if config.debug:
        _print_debug_summary(result)

    return 0


def apply_cli_overrides(config: TranscriptionConfig, args: argparse.Namespace) -> TranscriptionConfig:
    updated = config

    if args.min_fret is not None:
        updated.fretboard = replace(updated.fretboard, min_fret=args.min_fret)
    if args.max_fret is not None:
        updated.fretboard = replace(updated.fretboard, max_fret=args.max_fret)
    if args.confidence_threshold is not None:
        updated.pitch = replace(updated.pitch, confidence_threshold=args.confidence_threshold)
    if args.min_note_ms is not None:
        updated.notes = replace(updated.notes, min_note_ms=args.min_note_ms)
    if args.device is not None:
        updated.pitch = replace(updated.pitch, device=args.device)
    if args.debug:
        updated.debug = True

    return updated


def _print_debug_summary(result) -> None:
    import sys

    print(
        f"Detected {len(result.note_events)} notes over {result.duration_s:.2f}s "
        f"with total path cost {result.optimization.total_cost:.3f}.",
        file=sys.stderr,
    )
    for decision in result.optimization.decisions:
        print(
            f"{decision.note.start_s:6.3f}-{decision.note.end_s:6.3f}s "
            f"{decision.note.note_name:<4} conf={decision.note.mean_confidence:.2f} "
            f"-> string {decision.position.string_number} fret {decision.position.fret} "
            f"(local={decision.intrinsic_cost + decision.transition_cost:.2f})",
            file=sys.stderr,
        )
