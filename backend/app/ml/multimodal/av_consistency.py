import os
from typing import Dict, Any, List

import cv2
import librosa
import numpy as np


def calculate_audio_energy(
    audio_path: str,
    sample_rate: int = 16000,
    hop_length: int = 512
) -> np.ndarray:

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True
    )

    # RMS energy represents short-term audio intensity.
    rms = librosa.feature.rms(
        y=audio,
        hop_length=hop_length
    )[0]

    # Normalize to 0-1
    rms_min = np.min(rms)
    rms_max = np.max(rms)

    if rms_max - rms_min > 0:
        rms = (
            (rms - rms_min)
            / (rms_max - rms_min)
        )
    else:
        rms = np.zeros_like(rms)

    return rms


def calculate_mouth_activity(
    frame_results: List[Dict[str, Any]]
) -> Dict[str, Any]:

    timestamps = []
    mouth_activity = []

    for frame in frame_results:

        timestamp = frame.get("timestamp")
        faces = frame.get("faces", [])

        if timestamp is None:
            continue

        if not faces:
            continue

        # Use the largest detected face.
        face = max(
            faces,
            key=lambda item:
            item["width"] * item["height"]
        )

        bbox = face["bbox"]

        frame_path = frame.get("frame_path")

        if not frame_path or not os.path.exists(frame_path):
            continue

        image = cv2.imread(frame_path)

        if image is None:
            continue

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width <= 0 or face_height <= 0:
            continue

        # Approximate lower-face / mouth region.
        mouth_y1 = y1 + int(face_height * 0.55)
        mouth_y2 = y1 + int(face_height * 0.90)

        mouth_region = image[
            mouth_y1:mouth_y2,
            x1:x2
        ]

        if mouth_region.size == 0:
            continue

        # Convert to grayscale.
        gray = cv2.cvtColor(
            mouth_region,
            cv2.COLOR_BGR2GRAY
        )

        # Estimate local activity using edge density.
        edges = cv2.Canny(
            gray,
            threshold1=50,
            threshold2=150
        )

        activity = np.mean(edges > 0)

        timestamps.append(float(timestamp))
        mouth_activity.append(float(activity))

    if not mouth_activity:
        return {
            "timestamps": [],
            "mouth_activity": [],
            "samples": 0
        }

    values = np.array(
        mouth_activity,
        dtype=np.float32
    )

    # Normalize
    minimum = np.min(values)
    maximum = np.max(values)

    if maximum - minimum > 0:
        values = (
            (values - minimum)
            / (maximum - minimum)
        )
    else:
        values = np.zeros_like(values)

    return {
        "timestamps": timestamps,
        "mouth_activity": values.tolist(),
        "samples": len(values)
    }


def calculate_av_consistency(
    audio_path: str,
    frame_results: List[Dict[str, Any]],
    sample_rate: int = 16000,
    hop_length: int = 512
) -> Dict[str, Any]:

    print(
        "[DeepScan] Calculating audio-visual consistency..."
    )

    # ----------------------------------------
    # Audio activity
    # ----------------------------------------

    audio_energy = calculate_audio_energy(
        audio_path=audio_path,
        sample_rate=sample_rate,
        hop_length=hop_length
    )

    audio_duration = (
        len(audio_energy)
        * hop_length
        / sample_rate
    )

    # ----------------------------------------
    # Visual mouth activity
    # ----------------------------------------

    visual_result = calculate_mouth_activity(
        frame_results
    )

    visual_timestamps = np.array(
        visual_result["timestamps"],
        dtype=np.float32
    )

    visual_activity = np.array(
        visual_result["mouth_activity"],
        dtype=np.float32
    )

    if len(visual_activity) < 2:
        return {
            "status": "insufficient_data",
            "audio_samples": int(
                len(audio_energy)
            ),
            "visual_samples": int(
                len(visual_activity)
            ),
            "consistency_score": None
        }

    # ----------------------------------------
    # Interpolate audio activity at
    # visual timestamps
    # ----------------------------------------

    audio_timestamps = np.arange(
        len(audio_energy)
    ) * hop_length / sample_rate

    interpolated_audio = np.interp(
        visual_timestamps,
        audio_timestamps,
        audio_energy
    )

    # ----------------------------------------
    # Correlation
    # ----------------------------------------

    if (
        np.std(interpolated_audio) == 0
        or np.std(visual_activity) == 0
    ):
        correlation = 0.0

    else:
        correlation = np.corrcoef(
            interpolated_audio,
            visual_activity
        )[0, 1]

        if np.isnan(correlation):
            correlation = 0.0

    # Convert [-1, 1] → [0, 1]
    consistency_score = (
        float(correlation) + 1.0
    ) / 2.0

    print(
        "[DeepScan] AV correlation:",
        round(float(correlation), 4)
    )

    print(
        "[DeepScan] AV consistency score:",
        round(consistency_score, 4)
    )

    return {
        "status": "success",
        "audio_samples": int(
            len(audio_energy)
        ),
        "visual_samples": int(
            len(visual_activity)
        ),
        "audio_duration_seconds": round(
            audio_duration,
            3
        ),
        "correlation": round(
            float(correlation),
            4
        ),
        "consistency_score": round(
            consistency_score,
            4
        ),
        "visual_timestamps": (
            visual_timestamps.tolist()
        ),
        "audio_activity": (
            interpolated_audio.tolist()
        ),
        "visual_activity": (
            visual_activity.tolist()
        )
    }