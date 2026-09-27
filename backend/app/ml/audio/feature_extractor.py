import os
from typing import Dict, Any

import librosa
import numpy as np


def extract_mel_spectrogram(
    audio_path: str,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128
) -> Dict[str, Any]:

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("[DeepScan] Loading audio...")

    audio, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file contains no samples.")

    print(
        f"[DeepScan] Audio loaded. "
        f"Samples: {len(audio)}, Sample rate: {sr}"
    )

    # Generate Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    # Convert power spectrogram to decibel scale
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize
    mel_mean = np.mean(mel_db)
    mel_std = np.std(mel_db)

    if mel_std > 0:
        mel_normalized = (
            (mel_db - mel_mean) / mel_std
        )
    else:
        mel_normalized = mel_db - mel_mean

    print(
        f"[DeepScan] Mel spectrogram generated. "
        f"Shape: {mel_normalized.shape}"
    )

    return {
        "sample_rate": sr,
        "duration_seconds": round(
            len(audio) / sr,
            3
        ),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "mel_shape": list(mel_normalized.shape),
        "mel_mean": float(mel_mean),
        "mel_std": float(mel_std),
        "mel_spectrogram": mel_normalized.astype(
            np.float32
        )
    }