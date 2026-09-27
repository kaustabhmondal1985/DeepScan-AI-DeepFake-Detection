import os
from typing import Any, Dict, List

import cv2
from retinaface import RetinaFace


# ============================================================
# GLOBAL RETINAFACE MODEL
# ============================================================

_RETINAFACE_MODEL = None


# ============================================================
# RETINAFACE MODEL INITIALIZATION
# ============================================================

def get_retinaface_model():
    """
    Initialize RetinaFace model only once.

    The same model instance is reused for all frames.
    This avoids rebuilding the TensorFlow model for every frame.
    """

    global _RETINAFACE_MODEL

    if _RETINAFACE_MODEL is None:
        print(
            "[DeepScan] Initializing RetinaFace model..."
        )

        _RETINAFACE_MODEL = RetinaFace.build_model()

        print(
            "[DeepScan] RetinaFace model initialized."
        )

    return _RETINAFACE_MODEL


# ============================================================
# SINGLE FRAME FACE DETECTION
# ============================================================

def detect_faces(
    frame_path: str,
    save_crops: bool = True,
    crop_output_dir: str | None = None,
    crop_prefix: str = "frame"
) -> Dict[str, Any]:
    """
    Detect all faces in a single video frame using RetinaFace.

    The RetinaFace model is initialized once and reused
    across all frames.

    IMPORTANT:
    Face crop filenames are generated using the actual
    frame filename. This prevents different frames from
    overwriting the same crop file.

    Example:

        frame_000000.jpg
            -> frame_000000_face_00.jpg

        frame_000001.jpg
            -> frame_000001_face_00.jpg

        frame_000002.jpg
            -> frame_000002_face_00.jpg

    Args:
        frame_path:
            Path to extracted video frame.

        save_crops:
            Whether detected face crops should be saved.

        crop_output_dir:
            Directory where face crops are saved.

        crop_prefix:
            Optional prefix for crop filenames.
            The actual frame name is always included to
            guarantee uniqueness.

    Returns:
        Dictionary containing face detections and metadata.
    """

    # --------------------------------------------------------
    # 1. Validate frame path
    # --------------------------------------------------------

    if not os.path.exists(frame_path):
        raise FileNotFoundError(
            f"Frame not found: {frame_path}"
        )

    # --------------------------------------------------------
    # 2. Read frame
    # --------------------------------------------------------

    frame = cv2.imread(frame_path)

    if frame is None:
        raise ValueError(
            f"Unable to read frame: {frame_path}"
        )

    # --------------------------------------------------------
    # 3. Get shared RetinaFace model
    # --------------------------------------------------------

    model = get_retinaface_model()

    # --------------------------------------------------------
    # 4. Run RetinaFace
    # --------------------------------------------------------

    detections = RetinaFace.detect_faces(
        frame,
        threshold=0.9,
        model=model
    )

    faces: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # 5. Handle no detections
    # --------------------------------------------------------

    if (
        not detections
        or not isinstance(detections, dict)
    ):
        return {
            "frame_path": frame_path,
            "face_count": 0,
            "faces": []
        }

    # --------------------------------------------------------
    # 6. Create crop directory
    # --------------------------------------------------------

    if save_crops:

        if crop_output_dir is None:
            crop_output_dir = os.path.join(
                os.path.dirname(frame_path),
                "face_crops"
            )

        os.makedirs(
            crop_output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # 7. Get frame dimensions
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # 8. Get actual frame filename
    # --------------------------------------------------------

    frame_filename = os.path.basename(
        frame_path
    )

    frame_name = os.path.splitext(
        frame_filename
    )[0]

    # --------------------------------------------------------
    # 9. Process detected faces
    # --------------------------------------------------------

    for face_index, detection in enumerate(
        detections.values()
    ):

        if not isinstance(
            detection,
            dict
        ):
            continue

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        facial_area = detection.get(
            "facial_area"
        )

        confidence = detection.get(
            "score",
            0.0
        )

        if not facial_area:
            continue

        if len(facial_area) != 4:
            continue

        # ----------------------------------------------------
        # Convert bounding box to integers
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Clamp bounding box
        # ----------------------------------------------------

        x1 = max(
            0,
            min(
                x1,
                width - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                height - 1
            )
        )

        x2 = max(
            0,
            min(
                x2,
                width
            )
        )

        y2 = max(
            0,
            min(
                y2,
                height
            )
        )

        # ----------------------------------------------------
        # Validate bounding box
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:
            continue

        # ----------------------------------------------------
        # Face dimensions
        # ----------------------------------------------------

        face_width = x2 - x1
        face_height = y2 - y1

        # ----------------------------------------------------
        # Extract face crop
        # ----------------------------------------------------

        face_crop = frame[
            y1:y2,
            x1:x2
        ]

        if face_crop.size == 0:
            continue

        crop_path = None

        # ----------------------------------------------------
        # Save face crop
        # ----------------------------------------------------

        if save_crops:

            # IMPORTANT FIX:
            #
            # Old:
            #     frame_face_00.jpg
            #
            # This caused every video frame to overwrite
            # the previous frame's face crop.
            #
            # New:
            #     frame_000000_face_00.jpg
            #     frame_000001_face_00.jpg
            #     frame_000002_face_00.jpg
            #
            # Therefore every frame gets its own file.

            crop_filename = (
                f"{frame_name}"
                f"_face_{face_index:02d}.jpg"
            )

            crop_path = os.path.join(
                crop_output_dir,
                crop_filename
            )

            success = cv2.imwrite(
                crop_path,
                face_crop
            )

            if not success:
                raise RuntimeError(
                    f"Failed to save face crop: "
                    f"{crop_path}"
                )

        # ----------------------------------------------------
        # Store face information
        # ----------------------------------------------------

        face_info = {
            "face_index": face_index,
            "confidence": round(
                float(confidence),
                4
            ),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "width": face_width,
            "height": face_height,
            "crop_path": crop_path
        }

        faces.append(
            face_info
        )

    # --------------------------------------------------------
    # 10. Return result
    # --------------------------------------------------------

    return {
        "frame_path": frame_path,
        "face_count": len(faces),
        "faces": faces
    }


# ============================================================
# MULTIPLE FRAME FACE DETECTION
# ============================================================

def detect_faces_in_frames(
    frame_paths: List[str],
    save_crops: bool = True,
    crop_output_dir: str | None = None
) -> List[Dict[str, Any]]:
    """
    Detect faces in multiple video frames.

    RetinaFace is initialized only once and reused for
    all frames.

    Args:
        frame_paths:
            List of frame paths.

        save_crops:
            Whether face crops should be saved.

        crop_output_dir:
            Directory for face crops.

    Returns:
        List of detection results, one per frame.
    """

    results: List[Dict[str, Any]] = []

    total_frames = len(
        frame_paths
    )

    # --------------------------------------------------------
    # Initialize model once
    # --------------------------------------------------------

    get_retinaface_model()

    print(
        f"[DeepScan] RetinaFace will process "
        f"{total_frames} frames."
    )

    # --------------------------------------------------------
    # Process every frame
    # --------------------------------------------------------

    for index, frame_path in enumerate(
        frame_paths
    ):

        print(
            f"[DeepScan] RetinaFace processing "
            f"frame {index + 1}/{total_frames}"
        )

        try:

            result = detect_faces(
                frame_path=frame_path,
                save_crops=save_crops,
                crop_output_dir=crop_output_dir
            )

            results.append(
                result
            )

            print(
                f"[DeepScan] Frame {index + 1} "
                f"complete. Faces found: "
                f"{result.get('face_count', 0)}"
            )

        except Exception as exc:

            print(
                f"[DeepScan] Face detection failed "
                f"for frame {index + 1}: {exc}"
            )

            results.append(
                {
                    "frame_path": frame_path,
                    "face_count": 0,
                    "faces": [],
                    "error": str(exc)
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_faces = sum(
        result.get(
            "face_count",
            0
        )
        for result in results
    )

    print(
        f"[DeepScan] RetinaFace finished. "
        f"Total face detections: {total_faces}"
    )

    return results