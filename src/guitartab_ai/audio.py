from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import AudioConfig
from .models import AudioBuffer


def load_audio(path: str | Path, config: AudioConfig) -> AudioBuffer:
    """Load audio, convert to mono, normalize, and resample."""

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        if not config.mono:
            raise ValueError("Stereo processing is not supported for the MVP.")
        samples = np.mean(samples, axis=1, dtype=np.float32)

    if config.normalize_peak:
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0.0:
            samples = samples / peak

    if sample_rate != config.target_sample_rate:
        samples = librosa.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=config.target_sample_rate,
            res_type=config.resample_type,
        ).astype(np.float32)
        sample_rate = config.target_sample_rate

    return AudioBuffer(
        samples=samples.astype(np.float32, copy=False),
        sample_rate=sample_rate,
        duration_s=float(len(samples)) / float(sample_rate) if sample_rate else 0.0,
    )
