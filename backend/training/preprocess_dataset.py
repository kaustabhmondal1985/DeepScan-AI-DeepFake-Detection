from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
from retinaface import RetinaFace


# ============================================================
# PATHS
# ============================================================

SPLIT_DIR = Path("training/splits")
OUTPUT_ROOT = Path("training/processed")


# ============================================================
# CONFIGURATION
# ============================================================

FRAMES_PER_VIDEO = 8

# RetinaFace confidence is unreliable in the current
# environment, so this is recorded but NOT used as a
# hard filtering condition.
FACE_CONFIDENCE_THRESHOLD = 0.90

MIN_FACE_SIZE = 40

# Xception input size
CROP_SIZE = 299

# Margin around detected face
FACE_MARGIN = 0.20


# ============================================================
# FRAME SELECTION
# ============================================================

def select_frame_indices(
    total_frames: int,
    num_frames: int
):
    """
    Select evenly distributed frames across a video.
    """

    if total_frames <= 0:
        return []

    if total_frames <= num_frames:
        return list(range(total_frames))

    indices = np.linspace(
        0,
        total_frames - 1,
        num=num_frames,
        dtype=int
    )

    return sorted(
        set(indices.tolist())
    )


# ============================================================
# FACE DETECTION
# ============================================================

def detect_largest_face(frame):
    """
    Detect faces using RetinaFace.

    The current RetinaFace environment may return
    face_confidence=0.0 even when a valid face is detected.

    Therefore the bounding box is used as the primary
    validity check.

    If multiple faces are detected, the largest valid
    face is selected.
    """

    try:
        detections = RetinaFace.detect_faces(frame)

    except Exception as exc:
        print(
            f"      [WARNING] RetinaFace error: {exc}"
        )
        return None

    if not isinstance(detections, dict):
        return None

    candidates = []

    for detection in detections.values():

        if not isinstance(detection, dict):
            continue

        facial_area = detection.get(
            "facial_area"
        )

        if not facial_area or len(facial_area) != 4:
            continue

        try:
            x1, y1, x2, y2 = map(
                int,
                facial_area
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        width = x2 - x1
        height = y2 - y1

        if width < MIN_FACE_SIZE:
            continue

        if height < MIN_FACE_SIZE:
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        frame_height, frame_width = frame.shape[:2]

        x1 = max(
            0,
            min(
                x1,
                frame_width - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                frame_height - 1
            )
        )

        x2 = max(
            0,
            min(
                x2,
                frame_width
            )
        )

        y2 = max(
            0,
            min(
                y2,
                frame_height
            )
        )

        if x2 <= x1 or y2 <= y1:
            continue

        confidence = detection.get(
            "face_confidence"
        )

        if confidence is None:
            confidence = detection.get(
                "confidence"
            )

        if confidence is not None:

            try:
                confidence = float(
                    confidence
                )

            except (
                TypeError,
                ValueError
            ):
                confidence = None

        area = (
            x2 - x1
        ) * (
            y2 - y1
        )

        candidates.append(
            {
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                "confidence": confidence,
                "area": area
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# FACE CROPPING
# ============================================================

def crop_face(
    frame,
    bbox
):
    """
    Crop face with margin and resize to 299x299.
    """

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    margin_x = int(
        face_width * FACE_MARGIN
    )

    margin_y = int(
        face_height * FACE_MARGIN
    )

    x1 = max(
        0,
        x1 - margin_x
    )

    y1 = max(
        0,
        y1 - margin_y
    )

    x2 = min(
        frame_width,
        x2 + margin_x
    )

    y2 = min(
        frame_height,
        y2 + margin_y
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        (
            CROP_SIZE,
            CROP_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    return crop


# ============================================================
# RESUME CHECK
# ============================================================

def can_resume(
    metadata_path: Path
):
    """
    Reuse existing preprocessing only when it was created
    with the current FRAMES_PER_VIDEO configuration.
    """

    if not metadata_path.exists():
        return False

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            existing = json.load(file)

    except Exception:
        return False

    if existing.get("status") != "success":
        return False

    previous_frames = existing.get(
        "requested_frames"
    )

    if previous_frames != FRAMES_PER_VIDEO:
        return False

    if existing.get(
        "successful_face_crops",
        0
    ) <= 0:
        return False

    return True


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    row,
    split_name
):
    """
    Process one video.

    Video
       ↓
    Frame sampling
       ↓
    RetinaFace
       ↓
    Largest face
       ↓
    Crop
       ↓
    Resize 299x299
       ↓
    Save crop + metadata
    """

    video_path = Path(
        row["video_path"]
    )

    video_name = video_path.stem

    manipulation = row[
        "manipulation"
    ]

    output_dir = (
        OUTPUT_ROOT
        / split_name
        / f"{manipulation}_{video_name}"
    )

    faces_dir = (
        output_dir
        / "faces"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if can_resume(
        metadata_path
    ):

        print(
            f"      [SKIP] {video_name} "
            f"(already processed with "
            f"{FRAMES_PER_VIDEO} frames)"
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    if metadata_path.exists():

        print(
            f"      [REPROCESS] {video_name} "
            f"(different configuration)"
        )

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    faces_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    frame_indices = select_frame_indices(
        total_frames,
        FRAMES_PER_VIDEO
    )

    frame_results = []

    start_time = time.time()

    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    for position, frame_index in enumerate(
        frame_indices
    ):

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = capture.read()

        # ----------------------------------------------------
        # READ FAILURE
        # ----------------------------------------------------

        if not success:

            print(
                f"      Frame "
                f"{position + 1}/"
                f"{len(frame_indices)} "
                f"READ FAILED"
            )

            frame_results.append(
                {
                    "frame_index": int(
                        frame_index
                    ),
                    "timestamp": float(
                        frame_index / fps
                        if fps > 0
                        else 0
                    ),
                    "face_detected": False,
                    "crop_path": None,
                    "confidence": None,
                    "bbox": None
                }
            )

            continue

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0
        )

        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        face = detect_largest_face(
            frame
        )

        if face is None:

            print(
                f"      Frame "
                f"{position + 1}/"
                f"{len(frame_indices)} "
                f"NO FACE"
            )

            frame_results.append(
                {
                    "frame_index": int(
                        frame_index
                    ),
                    "timestamp": float(
                        timestamp
                    ),
                    "face_detected": False,
                    "crop_path": None,
                    "confidence": None,
                    "bbox": None
                }
            )

            continue

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        crop = crop_face(
            frame,
            face["bbox"]
        )

        if crop is None:

            print(
                f"      Frame "
                f"{position + 1}/"
                f"{len(frame_indices)} "
                f"INVALID CROP"
            )

            frame_results.append(
                {
                    "frame_index": int(
                        frame_index
                    ),
                    "timestamp": float(
                        timestamp
                    ),
                    "face_detected": False,
                    "crop_path": None,
                    "confidence": None,
                    "bbox": face["bbox"]
                }
            )

            continue

        # ----------------------------------------------------
        # SAVE CROP
        # ----------------------------------------------------

        crop_filename = (
            f"face_{position:02d}.jpg"
        )

        crop_path = (
            faces_dir
            / crop_filename
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        if face["confidence"] is not None:

            confidence_text = (
                f"{face['confidence']:.4f}"
            )

        else:

            confidence_text = "available"

        print(
            f"      Frame "
            f"{position + 1}/"
            f"{len(frame_indices)} "
            f"FACE "
            f"confidence={confidence_text}"
        )

        frame_results.append(
            {
                "frame_index": int(
                    frame_index
                ),
                "timestamp": float(
                    timestamp
                ),
                "face_detected": True,
                "crop_path": str(
                    crop_path
                ),
                "confidence": (
                    float(
                        face["confidence"]
                    )
                    if face["confidence"] is not None
                    else None
                ),
                "bbox": face["bbox"]
            }
        )

    capture.release()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    successful_crops = sum(
        1
        for item in frame_results
        if item["face_detected"]
    )

    processing_time = (
        time.time()
        - start_time
    )

    status = (
        "success"
        if successful_crops > 0
        else "failed"
    )

    metadata = {
        "status": status,
        "video_path": str(video_path),
        "video_name": video_name,
        "split": split_name,
        "label": int(row["label"]),
        "label_name": row["label_name"],
        "manipulation": manipulation,
        "source_group": row["source_group"],
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": round(
            duration,
            3
        ),
        "requested_frames": len(
            frame_indices
        ),
        "successful_face_crops": (
            successful_crops
        ),
        "processing_time_seconds": round(
            processing_time,
            3
        ),
        "frames": frame_results
    }

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    return metadata


# ============================================================
# PROCESS COMPLETE SPLIT
# ============================================================

def process_split(
    split_name
):
    """
    Process every video in the requested split.
    """

    csv_path = (
        SPLIT_DIR
        / f"{split_name}.csv"
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Split file not found: "
            f"{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"PROCESSING {split_name.upper()} "
        f"({len(df)} VIDEOS)"
    )

    print(
        "=" * 70
    )

    print(
        f"Frames per video: "
        f"{FRAMES_PER_VIDEO}"
    )

    start = time.time()

    results = []

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for index, row in df.iterrows():

        print(
            f"\n[{index + 1}/{len(df)}] "
            f"{row['video_name']} "
            f"| {row['label_name']} "
            f"| {row['manipulation']}"
        )

        try:

            result = process_video(
                row,
                split_name
            )

            results.append(
                result
            )

        except KeyboardInterrupt:

            print(
                "\n\n[INTERRUPTED] "
                "Processing stopped by user."
            )

            print(
                "Already completed videos "
                "will be skipped when restarted."
            )

            raise

        except Exception as exc:

            print(
                f"      [ERROR] {exc}"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start
    )

    successful = sum(
        1
        for result in results
        if result["status"] == "success"
    )

    failed = (
        len(results)
        - successful
    )

    crops = sum(
        result.get(
            "successful_face_crops",
            0
        )
        for result in results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{split_name.upper()} COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Videos: "
        f"{len(results)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Face crops: "
        f"{crops}"
    )

    print(
        f"Time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DeepScan - VALIDATION + TEST PREPROCESSING"
    )

    print(
        "=" * 70
    )

    # TRAIN is already complete.
    #
    # Process validation first.
    process_split("val")

    # Then process test.
    process_split("test")

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION + TEST PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()