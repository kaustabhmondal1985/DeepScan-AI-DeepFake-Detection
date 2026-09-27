import os
import json
from typing import Dict, Any

from app.ml.preprocessing.video_processor import (
    get_video_metadata
)

from app.ml.preprocessing.frame_extractor import (
    extract_frames
)

from app.ml.preprocessing.face_detector import (
    detect_faces_in_frames
)

from app.ml.preprocessing.face_tracker import (
    track_faces
)


def preprocess_video(
    video_path: str,
    output_dir: str
) -> Dict[str, Any]:

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        f"[DeepScan] Processing video: "
        f"{video_path}"
    )

    # =====================================================
    # 1. VIDEO METADATA
    # =====================================================

    metadata = get_video_metadata(
        video_path
    )

    print(
        "[DeepScan] Video metadata:",
        metadata
    )

    # =====================================================
    # 2. FRAME EXTRACTION
    # =====================================================

    frames_dir = os.path.join(
        output_dir,
        "frames"
    )

    frame_data = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        sample_fps=5.0,
        max_frames=300
    )

    print(
        f"[DeepScan] Extracted "
        f"{frame_data['frames_extracted']} frames"
    )

    # =====================================================
    # 3. FACE DETECTION
    # =====================================================

    print(
        "[DeepScan] Starting RetinaFace detection..."
    )

    face_detection_result = (
        detect_faces_in_frames(
            frames=frame_data["frames"],
            output_dir=output_dir
        )
    )

    print(
        f"[DeepScan] Face detection complete. "
        f"Processed "
        f"{face_detection_result['frames_processed']} "
        f"frames and found "
        f"{face_detection_result['total_face_detections']} "
        f"face detections."
    )

    # =====================================================
    # 4. FACE TRACKING
    # =====================================================

    print(
        "[DeepScan] Starting face tracking..."
    )

    tracking_result = track_faces(
        frame_results=
            face_detection_result["frames"]
    )

    print(
        f"[DeepScan] Face tracking complete. "
        f"Found "
        f"{tracking_result['total_tracks']} "
        f"face tracks."
    )

    # =====================================================
    # 5. BUILD PREPROCESSING RESULT
    # =====================================================

    preprocessing_result = {

        "video": metadata,

        "frames": {
            "sample_fps":
                frame_data["sample_fps"],

            "frames_extracted":
                frame_data["frames_extracted"]
        },

        "face_detection": {

            "frames_processed":
                face_detection_result[
                    "frames_processed"
                ],

            "total_face_detections":
                face_detection_result[
                    "total_face_detections"
                ],

            "frames":
                face_detection_result[
                    "frames"
                ]
        },

        "face_tracking": {

            "total_tracks":
                tracking_result[
                    "total_tracks"
                ],

            "tracks":
                tracking_result[
                    "tracks"
                ]
        }
    }

    # =====================================================
    # 6. SAVE PREPROCESSING JSON
    # =====================================================

    metadata_path = os.path.join(
        output_dir,
        "preprocessing.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            preprocessing_result,
            file,
            indent=4
        )

    print(
        f"[DeepScan] Preprocessing complete: "
        f"{metadata_path}"
    )

    return preprocessing_result