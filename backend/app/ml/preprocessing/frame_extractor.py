import os
import cv2
from typing import List, Dict, Any


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_fps: float = 5.0,
    max_frames: int = 300
) -> Dict[str, Any]:
    """
    Extract sampled frames from a video.

    By default:
        1 frame per second
        maximum 300 frames
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        capture.release()
        raise ValueError(
            "Unable to determine video FPS."
        )

    # Number of source frames to skip
    frame_interval = max(
        int(round(fps / sample_fps)),
        1
    )

    frame_index = 0
    extracted_count = 0

    extracted_frames: List[Dict[str, Any]] = []

    while extracted_count < max_frames:

        success, frame = capture.read()

        if not success:
            break

        if frame_index % frame_interval == 0:

            timestamp = frame_index / fps

            filename = (
                f"frame_{extracted_count:06d}.jpg"
            )

            frame_path = os.path.join(
                output_dir,
                filename
            )

            saved = cv2.imwrite(
                frame_path,
                frame
            )

            if saved:

                extracted_frames.append({
                    "index": extracted_count,
                    "source_frame": frame_index,
                    "timestamp": round(
                        timestamp,
                        3
                    ),
                    "path": frame_path
                })

                extracted_count += 1

        frame_index += 1

    capture.release()

    return {
        "sample_fps": sample_fps,
        "frames_extracted": extracted_count,
        "frames": extracted_frames
    }