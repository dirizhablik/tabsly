from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.signal import medfilt

from .config import PitchConfig
from .models import AudioBuffer, PitchFrame
from .music import hz_to_midi, quantize_midi


class PitchEstimator(Protocol):
    """Interface for frame-level pitch estimators."""

    def estimate(self, audio: AudioBuffer, config: PitchConfig) -> list[PitchFrame]:
        ...


class PitchEstimatorError(RuntimeError):
    """Raised when the configured pitch backend cannot run."""


@dataclass(slots=True)
class TorchCrepePitchEstimator:
    """Neural F0 estimation using torchcrepe."""

    def estimate(self, audio: AudioBuffer, config: PitchConfig) -> list[PitchFrame]:
        try:
            import torch
            import torchcrepe
        except ImportError as exc:
            raise PitchEstimatorError(
                "torchcrepe backend is unavailable. Install `torch` and `torchcrepe`."
            ) from exc

        if audio.sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive.")

        device = _resolve_device(config.device, torch.cuda.is_available())
        hop_length = max(1, int(round((config.frame_hop_ms / 1_000.0) * audio.sample_rate)))
        kernel_size = _odd_kernel_width(config.median_filter_width)

        waveform = torch.from_numpy(audio.samples).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            predicted = torchcrepe.predict(
                waveform,
                audio.sample_rate,
                hop_length,
                config.fmin_hz,
                config.fmax_hz,
                model=config.model_capacity,
                batch_size=config.batch_size,
                device=device,
                decoder=torchcrepe.decode.viterbi,
                return_periodicity=True,
            )

        if isinstance(predicted, tuple):
            pitch_tensor, confidence_tensor = predicted
        else:
            pitch_tensor = predicted
            confidence_tensor = torch.ones_like(pitch_tensor)

        frequencies = pitch_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        confidences = confidence_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)

        if kernel_size > 1 and frequencies.size >= kernel_size:
            frequencies = medfilt(frequencies, kernel_size=kernel_size)
            confidences = medfilt(confidences, kernel_size=kernel_size)

        hop_s = hop_length / float(audio.sample_rate)
        frames: list[PitchFrame] = []
        for index, (frequency_hz, confidence) in enumerate(zip(frequencies, confidences, strict=True)):
            time_s = index * hop_s
            raw_frequency = float(frequency_hz) if np.isfinite(frequency_hz) and frequency_hz > 0.0 else None
            midi_continuous = hz_to_midi(raw_frequency) if raw_frequency is not None else None
            voiced = bool(raw_frequency is not None and confidence >= config.confidence_threshold)
            midi_rounded = quantize_midi(midi_continuous) if voiced and midi_continuous is not None else None
            frames.append(
                PitchFrame(
                    index=index,
                    time_s=time_s,
                    frequency_hz=raw_frequency,
                    confidence=float(confidence),
                    midi_continuous=midi_continuous,
                    midi_rounded=midi_rounded,
                    voiced=voiced,
                )
            )
        return frames


def build_pitch_estimator(config: PitchConfig) -> PitchEstimator:
    """Instantiate the configured pitch backend."""

    if config.backend == "torchcrepe":
        return TorchCrepePitchEstimator()
    raise ValueError(f"Unsupported pitch backend: {config.backend}")


def _resolve_device(device: str, cuda_available: bool) -> str:
    if device == "auto":
        return "cuda" if cuda_available else "cpu"
    return device


def _odd_kernel_width(width: int) -> int:
    if width <= 1:
        return 1
    return width if width % 2 == 1 else width + 1
