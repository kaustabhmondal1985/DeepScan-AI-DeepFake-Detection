from pathlib import Path
import cv2
from retinaface import RetinaFace


VIDEO_PATH = Path(
    r"C:\Users\Kaustabh\Downloads\FaceForensics++_test\original_sequences\youtube\c23\videos\033.mp4"
)


def main():

    print("=" * 70)
    print("DeepScan - RetinaFace Training Video Diagnostic")
    print("=" * 70)

    print(f"\nVideo:")
    print(VIDEO_PATH)

    if not VIDEO_PATH.exists():
        print("\nERROR: Video does not exist.")
        return

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        print("\nERROR: Could not open video.")
        return

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("\nVideo metadata:")
    print(f"  Frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")

    # Test 4 evenly distributed frames
    if total_frames > 4:

        frame_indices = [
            0,
            total_frames // 3,
            (2 * total_frames) // 3,
            total_frames - 1
        ]

    else:

        frame_indices = list(
            range(total_frames)
        )

    print(
        f"\nTesting frames: {frame_indices}"
    )

    for position, frame_index in enumerate(
        frame_indices
    ):

        print(
            f"\n--- Frame "
            f"{position + 1}/{len(frame_indices)} "
            f"(source frame {frame_index}) ---"
        )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = capture.read()

        if not success:

            print("Could not read frame.")
            continue

        print(
            f"Frame shape: {frame.shape}"
        )

        print(
            "Running RetinaFace..."
        )

        detections = RetinaFace.detect_faces(
            frame
        )

        if not isinstance(
            detections,
            dict
        ):

            print(
                "RESULT: NO DETECTIONS"
            )

            continue

        print(
            f"RESULT: {len(detections)} "
            f"face(s) detected"
        )

        for key, detection in detections.items():

            confidence = detection.get(
                "face_confidence",
                0.0
            )

            bbox = detection.get(
                "facial_area"
            )

            print(
                f"  {key}: "
                f"confidence={confidence:.4f}, "
                f"bbox={bbox}"
            )

    capture.release()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()