import os
import json
import time

import cv2
from retinaface import RetinaFace


FRAMES_JSON = "outputs/av_test/frames.json"


def main():

    print("=" * 60)
    print("DeepScan - RetinaFace 5-Frame Benchmark")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load frame metadata
    # --------------------------------------------------

    print("\n[1] Loading frame metadata...")

    with open(FRAMES_JSON, "r") as f:
        data = json.load(f)

    frames = data["frames"]

    print(
        f"[DeepScan] Total available frames: "
        f"{len(frames)}"
    )

    # Only test first 5 frames
    test_frames = frames[:5]

    print(
        f"[DeepScan] Testing only "
        f"{len(test_frames)} frames."
    )

    # --------------------------------------------------
    # 2. Process frames
    # --------------------------------------------------

    results = []

    print("\n[2] Running RetinaFace...")

    for i, frame in enumerate(test_frames):

        frame_path = frame["path"]
        timestamp = frame["timestamp"]

        print("\n" + "-" * 50)
        print(
            f"Processing frame {i + 1}/5"
        )
        print(
            f"Frame index: {frame['index']}"
        )
        print(
            f"Timestamp: {timestamp:.3f}s"
        )
        print(
            f"Path: {frame_path}"
        )

        if not os.path.exists(frame_path):

            print(
                "[ERROR] Frame file does not exist."
            )

            continue

        image = cv2.imread(frame_path)

        if image is None:

            print(
                "[ERROR] OpenCV could not read frame."
            )

            continue

        print(
            f"Image shape: {image.shape}"
        )

        # ----------------------------------------------
        # RetinaFace timing
        # ----------------------------------------------

        start_time = time.perf_counter()

        detections = RetinaFace.detect_faces(
            image
        )

        elapsed = (
            time.perf_counter() - start_time
        )

        # ----------------------------------------------
        # Count faces
        # ----------------------------------------------

        face_count = 0

        if isinstance(detections, dict):
            face_count = len(detections)

        print(
            f"RetinaFace time: "
            f"{elapsed:.2f} seconds"
        )

        print(
            f"Faces detected: "
            f"{face_count}"
        )

        # ----------------------------------------------
        # Print detections
        # ----------------------------------------------

        if face_count > 0:

            for face_id, detection in detections.items():

                if not isinstance(detection, dict):
                    continue

                print(
                    f"  Face: {face_id}"
                )

                print(
                    f"  Confidence: "
                    f"{detection.get('score', 0):.4f}"
                )

                print(
                    f"  BBox: "
                    f"{detection.get('facial_area')}"
                )

        results.append(
            {
                "frame_index": frame["index"],
                "timestamp": timestamp,
                "faces": face_count,
                "inference_seconds": round(
                    elapsed,
                    3
                )
            }
        )

    # --------------------------------------------------
    # 3. Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RETINAFACE BENCHMARK RESULT")
    print("=" * 60)

    successful = len(results)

    total_faces = sum(
        result["faces"]
        for result in results
    )

    total_time = sum(
        result["inference_seconds"]
        for result in results
    )

    print(
        f"Frames successfully processed: "
        f"{successful}/5"
    )

    print(
        f"Total faces detected: "
        f"{total_faces}"
    )

    print(
        f"Total inference time: "
        f"{total_time:.2f} seconds"
    )

    if successful > 0:

        print(
            f"Average inference time: "
            f"{total_time / successful:.2f} seconds/frame"
        )

    print("\n" + "=" * 60)
    print("5-FRAME RETINAFACE TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()