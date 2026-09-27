import json
import os
import subprocess
from typing import Dict, Any


def has_audio_stream(video_path: str) -> bool:
    """
    Check whether the input video contains an audio stream.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        video_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFprobe failed while checking audio stream:\n"
            + result.stderr
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            "Could not parse FFprobe output."
        )

    streams = data.get("streams", [])

    return len(streams) > 0


def extract_audio(
    video_path: str,
    output_dir: str,
    sample_rate: int = 16000
) -> Dict[str, Any]:

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        "[DeepScan] Checking for audio stream..."
    )

    audio_available = has_audio_stream(
        video_path
    )

    # --------------------------------------------------
    # CASE 1: Video has no audio
    # --------------------------------------------------

    if not audio_available:

        print(
            "[DeepScan] No audio stream found."
        )

        return {
            "status": "unavailable",
            "reason": (
                "Input video does not contain "
                "an audio stream."
            ),
            "audio_path": None,
            "sample_rate": None,
            "channels": None,
            "format": None,
            "codec": None,
            "file_size_bytes": 0
        }

    # --------------------------------------------------
    # CASE 2: Audio exists
    # --------------------------------------------------

    audio_path = os.path.join(
        output_dir,
        "audio.wav"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        audio_path
    ]

    print(
        "[DeepScan] Audio stream found."
    )

    print(
        "[DeepScan] Extracting audio..."
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg audio extraction failed:\n"
            + result.stderr
        )

    if not os.path.exists(audio_path):
        raise RuntimeError(
            "Audio extraction completed but "
            "output file was not created."
        )

    file_size = os.path.getsize(
        audio_path
    )

    print(
        f"[DeepScan] Audio extracted: "
        f"{audio_path}"
    )

    return {
        "status": "success",
        "reason": None,
        "audio_path": audio_path,
        "sample_rate": sample_rate,
        "channels": 1,
        "format": "wav",
        "codec": "pcm_s16le",
        "file_size_bytes": file_size
    }