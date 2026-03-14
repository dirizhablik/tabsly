# GuitarTab AI

GuitarTab AI is a production-minded MVP for converting clean monophonic guitar audio into playable ASCII tablature. The system is intentionally split into replaceable modules so the pitch engine, note extraction logic, fretboard model, and fingering optimizer can evolve independently into a larger ML transcription product.

This MVP solves a specific, realistic problem well:

- Input: clean single-note guitar audio
- Output: note sequence plus globally optimized, playable guitar tab
- Core intelligence: neural pitch tracking plus dynamic-programming fingering search

## A. Executive Technical Plan

### Product goal

Build a technically serious transcription pipeline for clean monophonic guitar phrases:

1. Load audio and normalize it for inference.
2. Run a strong neural pitch estimator.
3. Convert frame-level pitch to stable symbolic note events.
4. Remove pitch jitter and transient segmentation noise.
5. Generate all valid fretboard positions for each note.
6. Use dynamic programming to choose the lowest-cost playable path over the entire phrase.
7. Render the chosen path as readable ASCII tablature.

### Pitch engine decision

This MVP uses `torchcrepe` as the default backend.

Why `torchcrepe` instead of Basic Pitch:

- `torchcrepe` is optimized for accurate frame-level monophonic F0 tracking, which is exactly the information needed for robust note cleanup and downstream fingering optimization.
- Basic Pitch is stronger when the task is direct multi-note transcription or mixed/polyphonic note extraction, but it is heavier and less controllable for a clean single-note guitar MVP where precise F0 contours matter more than end-to-end note decoding.
- `torchcrepe` gives both pitch and periodicity/confidence, which makes silence filtering and confidence-aware cleanup straightforward.

This is the strongest practical choice for the current scope.

### Pipeline summary

1. Audio loading
   - Read audio with `soundfile`
   - Downmix to mono
   - Peak-normalize
   - Resample to 16 kHz
2. Neural pitch inference
   - Run `torchcrepe` at 10 ms hop size
   - Decode with Viterbi smoothing
   - Keep periodicity as frame confidence
3. Note extraction
   - Convert Hz to continuous MIDI
   - Median-smooth voiced segments
   - Quantize to nearest semitone
   - Bridge short unvoiced gaps
   - Suppress short pitch runs caused by vibrato or jitter
   - Drop or merge very short note events
4. Fretboard mapping
   - Enumerate all string/fret positions for standard tuning within configured fret limits
5. Fingering optimization
   - Each note becomes a layer of candidate states
   - Dynamic programming chooses the minimum-cost global path
6. Rendering
   - Produce six-line ASCII tab
   - Optionally append a note listing with timestamps and chosen positions

## B. Architecture And Module Breakdown

### Input assumptions

- Clean monophonic guitar recording
- One fundamental pitch at a time
- Standard tuning: `E2 A2 D3 G3 B3 E4`
- No drums, vocals, or dense accompaniment
- No full polyphony or chord decoding in this MVP

### Preprocessing

Implemented in [src/guitartab_ai/audio.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/audio.py).

- Stereo is collapsed to mono by averaging channels.
- Peak normalization standardizes inference conditions.
- Audio is resampled to 16 kHz because `torchcrepe` operates well there and the pitch range is easily covered.

### Frame-level pitch inference

Implemented in [src/guitartab_ai/pitch.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/pitch.py).

- Backend: `torchcrepe`
- Hop size: 10 ms
- Pitch range: 70 Hz to 1400 Hz
- Decoder: Viterbi
- Confidence: periodicity score from `torchcrepe`
- Smoothing: median filter over pitch and confidence outputs

### Note extraction and denoising

Implemented in [src/guitartab_ai/notes.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/notes.py).

Frame sequence to notes:

1. Keep only voiced frames above confidence threshold.
2. Convert voiced Hz to fractional MIDI.
3. Median-smooth continuous MIDI values inside voiced runs.
4. Quantize to nearest semitone.
5. Bridge short silent gaps if the note on both sides is the same.
6. Suppress short note runs below `pitch_stability_frames`.
7. Convert cleaned runs into `NoteEvent` objects.
8. Remove or merge short note events below `min_note_ms`.

Vibrato handling for MVP:

- No separate vibrato classifier is used.
- Vibrato is handled heuristically by smoothing continuous MIDI and suppressing short, unstable semitone flips.
- This works well for clean monophonic audio where vibrato stays centered around a stable pitch.
- It will not correctly preserve expressive pitch ornaments yet. That is a roadmap item.

### Fretboard model

Implemented in [src/guitartab_ai/fretboard.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/fretboard.py).

- Standard tuned six-string guitar
- Configurable fret range, default `0-20`
- For every note, generate every valid string/fret position
- Deterministic ordering so optimization remains reproducible

### Optimization/search

Implemented in [src/guitartab_ai/optimize.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/optimize.py).

Each note is a time step. Each playable string/fret candidate is a state. The optimizer computes the minimum cumulative cost path with classic dynamic programming.

Cost terms:

- Intrinsic position cost
  - Slight bias toward lower frets
  - Additional penalty above fret 12
  - Small open-string bonus
- Transition cost
  - Absolute fret shift penalty
  - String change penalty
  - Same-string motion penalty
  - Quadratic large-shift penalty after a threshold
  - Cross-string plus position-shift penalty
  - Penalty for changing position on repeated notes
  - Shift relief when one side of the transition is an open string
  - Time-based relief when the previous note lasts long enough to reposition

Why this shape works:

- Monophonic guitar playability is mostly about left-hand movement economy.
- Pure note correctness is already fixed by the pitch sequence, so the hard part is choosing a physically sensible path across redundant fretboard positions.
- A globally optimized cost beats greedy local choices because a locally cheap fingering can force an awkward later jump.

### ASCII tab rendering

Implemented in [src/guitartab_ai/render.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/render.py).

- Standard six-line tab order: `e B G D A E`
- Monospaced sequential rendering
- Preserves note order
- Optionally appends a readable note listing with timestamps and selected positions

### CLI and orchestration

- Pipeline orchestration: [src/guitartab_ai/pipeline.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/pipeline.py)
- CLI: [src/guitartab_ai/cli.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/cli.py)

## C. Concrete Repo Structure

```text
project_root/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  examples/
    default_config.toml
  src/
    guitartab_ai/
      __init__.py
      __main__.py
      audio.py
      cli.py
      config.py
      fretboard.py
      midi.py
      models.py
      music.py
      notes.py
      optimize.py
      pipeline.py
      pitch.py
      render.py
      transcribe.py
  tests/
    test_fretboard.py
    test_music.py
    test_notes.py
    test_optimize.py
    test_render.py
```

File responsibilities:

- [src/guitartab_ai/models.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/models.py): typed domain objects
- [src/guitartab_ai/config.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/config.py): nested dataclass configuration plus TOML loading
- [src/guitartab_ai/audio.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/audio.py): loading, mono conversion, normalization, resampling
- [src/guitartab_ai/pitch.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/pitch.py): `torchcrepe` pitch backend
- [src/guitartab_ai/notes.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/notes.py): note segmentation and cleanup
- [src/guitartab_ai/fretboard.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/fretboard.py): standard tuning fretboard and candidate generation
- [src/guitartab_ai/optimize.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/optimize.py): global fingering search
- [src/guitartab_ai/render.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/render.py): ASCII tab output
- [src/guitartab_ai/midi.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/midi.py): optional MIDI export
- [src/guitartab_ai/pipeline.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/pipeline.py): end-to-end orchestration
- [src/guitartab_ai/cli.py](/Users/sas/Desktop/TABSLY/src/guitartab_ai/cli.py): command line entrypoint

## D. Dependency And Setup Instructions

### Dependencies

Required runtime dependencies:

- `numpy`
- `scipy`
- `librosa`
- `soundfile`
- `torch`
- `torchcrepe`
- `pretty_midi`

Dev/test dependency:

- `pytest`

### Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If you prefer `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Run locally

```bash
python -m guitartab_ai.transcribe input.wav --out output.txt
```

Example with overrides:

```bash
python -m guitartab_ai.transcribe input.wav \
  --config examples/default_config.toml \
  --min-fret 0 \
  --max-fret 20 \
  --confidence-threshold 0.40 \
  --min-note-ms 90 \
  --emit-midi output.mid \
  --debug
```

### Run tests

```bash
pytest
```

## E. Full Source Code For The MVP

The full runnable code lives in:

- [src/guitartab_ai](/Users/sas/Desktop/TABSLY/src/guitartab_ai)
- [tests](/Users/sas/Desktop/TABSLY/tests)
- [examples/default_config.toml](/Users/sas/Desktop/TABSLY/examples/default_config.toml)

## F. Tests

Implemented test coverage:

- [tests/test_music.py](/Users/sas/Desktop/TABSLY/tests/test_music.py)
  - Hz to MIDI conversion
  - MIDI quantization
  - Note name formatting
- [tests/test_notes.py](/Users/sas/Desktop/TABSLY/tests/test_notes.py)
  - Short gap bridging
  - Jitter suppression
  - Short note removal
- [tests/test_fretboard.py](/Users/sas/Desktop/TABSLY/tests/test_fretboard.py)
  - Candidate position generation
- [tests/test_optimize.py](/Users/sas/Desktop/TABSLY/tests/test_optimize.py)
  - DP path continuity on toy phrases
  - Repeated note stability
- [tests/test_render.py](/Users/sas/Desktop/TABSLY/tests/test_render.py)
  - ASCII formatting placement

## G. README

This file is the primary README and includes the architecture, setup, usage, algorithm choices, and roadmap.

## H. Explanation Of Key Algorithmic Choices

### Why frame-level pitch first

For guitar tablature, a robust symbolic note stream matters more than an end-to-end MIDI guess. Using frame-level F0 keeps the intermediate representation inspectable and lets us explicitly control note cleanup.

### Why semitone quantization after smoothing

Quantizing too early causes vibrato and pitch jitter to fragment into false notes. Smoothing first produces a stable continuous contour, then semitone quantization turns that contour into symbolic note identities.

### Why dynamic programming instead of greedy fingering

Guitar fretboard redundancy means every note can often be played in multiple places. Greedy local selection is shortsighted. DP is the correct MVP algorithm because it is:

- exact for the defined cost function
- fast enough for realistic monophonic phrases
- easy to inspect and tune
- extensible to richer state later

### Why not model left-hand fingers explicitly yet

This MVP optimizes position path, not full finger assignment. That is deliberate. Full finger modeling increases state complexity and requires much richer constraints. Position-level DP captures most of the playability win for single-note phrases while keeping the system maintainable.

## I. Future ML Roadmap

### Dataset integration

- Integrate GuitarSet for aligned guitar audio and symbolic annotations.
- Build data loaders that convert annotation formats into note-event plus fingering supervision.
- Benchmark note event extraction against dataset-aligned ground truth.

### Supervised fingering prediction

- Learn a prior over candidate positions instead of relying purely on hand-crafted costs.
- Features: note pitch, interval context, local phrase direction, string history, fret history, note duration.
- Candidate models: gradient-boosted trees first, then sequence models over candidate graphs.
- Use the learned score as an additive term inside the existing DP objective rather than replacing optimization.

### Technique classification

- Add per-note articulation labels: slide, hammer-on, pull-off, bend, vibrato.
- Start with note-level classifiers over pitch contour derivatives, onset envelope, and spectral features.
- Later move to multitask models that jointly infer pitch, onset, and technique states.

### Human-in-the-loop correction learning

- Let users edit suggested tabs.
- Capture corrections as preference data over candidate paths.
- Learn cost reweighting or ranking models from accepted vs rejected fingerings.

### Product evolution

- Add MIDI and GuitarPro export
- Add phrase-level confidence reporting
- Add waveform plus pitch-debug visualization
- Add partial polyphony support
- Add a desktop or web review UI with editable tab lanes

## Limitations

- Clean monophonic audio only
- No chord or dense polyphony support
- No explicit bends, slides, hammer-ons, pull-offs, or vibrato labels
- Rhythm notation is intentionally approximate
- Fingering is position-aware but not full left-hand finger assignment
