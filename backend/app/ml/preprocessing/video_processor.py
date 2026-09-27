import os
import cv2
from typing import Dict, Any


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract basic metadata from a video.

    Returns:
        duration
        fps
        width
        height
        total_frames
        codec
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # Avoid division by zero
    duration = (
        total_frames / fps
        if fps and fps > 0
        else 0
    )

    codec_value = int(
        capture.get(cv2.CAP_PROP_FOURCC)
    )

    codec = "".join(
        [
            chr(codec_value & 0xFF),
            chr((codec_value >> 8) & 0xFF),
            chr((codec_value >> 16) & 0xFF),
            chr((codec_value >> 24) & 0xFF),
        ]
    )

    capture.release()

    return {
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "total_frames": total_frames,
        "codec": codec.strip()
    }