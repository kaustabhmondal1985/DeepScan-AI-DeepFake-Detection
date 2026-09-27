import os
import json
import time

import cv2
from retinaface import RetinaFace

from app.ml.multimodal.av_consistency import (
    calculate_av_consistency
)


VIDEO_PATH = "test_videos/talking_person.mp4"

OUTPUT_DIR = "outputs/av_talking_test"

AUDIO_PATH = os.path.join(
    OUTPUT_DIR,
    "audio.wav"
)

FRAME_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "face_results.json"
)


def extract_audio():

    print("\n" + "=" * 60)
    print("STEP 1 - AUDIO EXTRACTION")
    print("=" * 60)

    from app.ml.audio.audio_processor import (
        extract_audio as extract_audio_file
    )

    result = extract_audio_file(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=16000
    )

    print(
        f"\nAudio status: "
        f"{result['status']}"
    )

    print(
        f"Audio path: "
        f"{result['audio_path']}"
    )

    if result["status"] != "success":
        raise RuntimeError(
            "Audio extraction failed. "
            "AV consistency requires audio."
        )

    return result["audio_path"]


def extract_sampled_frames():

    print("\n" + "=" * 60)
    print("STEP 2 - SAMPLING VIDEO FRAMES")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    capture = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open input video."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    # ----------------------------------------------
    # We deliberately use only 5 frames.
    #
    # This is a validation test because RetinaFace
    # is currently slow on CPU.
    # ----------------------------------------------

    num_samples = 5

    if total_frames < num_samples:
        num_samples = total_frames

    frame_indices = [
        round(
            i * (total_frames - 1)
            / (num_samples - 1)
        )
        for i in range(num_samples)
    ]

    print(
        f"Sampling {len(frame_indices)} "
        f"frames:"
    )

    print(frame_indices)

    frames = []

    for sample_number, frame_index in enumerate(
        frame_indices
    ):

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = capture.read()

        if not success:
            print(
                f"[WARNING] Could not read "
                f"frame {frame_index}"
            )
            continue

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0
        )

        frame_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{sample_number:02d}.jpg"
        )

        cv2.imwrite(
            frame_path,
            frame
        )

        frames.append(
            {
                "index": sample_number,
                "source_frame": frame_index,
                "timestamp": float(timestamp),
                "path": frame_path
            }
        )

        print(
            f"Frame {sample_number + 1}/"
            f"{len(frame_indices)} "
            f"→ source frame {frame_index} "
            f"→ {timestamp:.3f}s"
        )

    capture.release()

    print(
        f"\nFrames successfully extracted: "
        f"{len(frames)}"
    )

    return frames


def run_retinaface(frames):

    print("\n" + "=" * 60)
    print("STEP 3 - RETINAFACE")
    print("=" * 60)

    frame_results = []

    total = len(frames)

    print(
        f"Running RetinaFace on "
        f"{total} sampled frames..."
    )

    for i, frame in enumerate(frames):

        print("\n" + "-" * 50)

        print(
            f"Processing frame "
            f"{i + 1}/{total}"
        )

        print(
            f"Timestamp: "
            f"{frame['timestamp']:.3f}s"
        )

        image = cv2.imread(
            frame["path"]
        )

        if image is None:
            print(
                "[WARNING] Could not read image."
            )
            continue

        start = time.perf_counter()

        detections = RetinaFace.detect_faces(
            image
        )

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"RetinaFace inference: "
            f"{elapsed:.2f}s"
        )

        faces = []

        if isinstance(
            detections,
            dict
        ):

            for face_id, detection in (
                detections.items()
            ):

                if not isinstance(
                    detection,
                    dict
                ):
                    continue

                facial_area = detection.get(
                    "facial_area"
                )

                if not facial_area:
                    continue

                x1, y1, x2, y2 = (
                    facial_area
                )

                height, width = (
                    image.shape[:2]
                )

                x1 = max(
                    0,
                    min(int(x1), width - 1)
                )

                y1 = max(
                    0,
                    min(int(y1), height - 1)
                )

                x2 = max(
                    0,
                    min(int(x2), width - 1)
                )

                y2 = max(
                    0,
                    min(int(y2), height - 1)
                )

                face_width = x2 - x1
                face_height = y2 - y1

                if (
                    face_width <= 0
                    or face_height <= 0
                ):
                    continue

                faces.append(
                    {
                        "face_id": face_id,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        },
                        "width": face_width,
                        "height": face_height,
                        "confidence": float(
                            detection.get(
                                "score",
                                0.0
                            )
                        )
                    }
                )

        print(
            f"Faces detected: "
            f"{len(faces)}"
        )

        for face in faces:

            print(
                f"  Confidence: "
                f"{face['confidence']:.4f}"
            )

            print(
                f"  BBox: "
                f"{face['bbox']}"
            )

        frame_results.append(
            {
                "frame_index": frame["index"],
                "timestamp": frame["timestamp"],
                "frame_path": frame["path"],
                "faces": faces
            }
        )

    print("\n" + "=" * 60)

    total_faces = sum(
        len(frame["faces"])
        for frame in frame_results
    )

    frames_with_faces = sum(
        1
        for frame in frame_results
        if frame["faces"]
    )

    print(
        f"Frames processed: "
        f"{len(frame_results)}"
    )

    print(
        f"Frames with faces: "
        f"{frames_with_faces}"
    )

    print(
        f"Total face detections: "
        f"{total_faces}"
    )

    return frame_results


def save_face_results(frame_results):

    output = {
        "frames_processed": len(
            frame_results
        ),
        "frames_with_faces": sum(
            1
            for frame in frame_results
            if frame["faces"]
        ),
        "total_face_detections": sum(
            len(frame["faces"])
            for frame in frame_results
        ),
        "frames": frame_results
    }

    with open(
        FRAME_RESULTS_PATH,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print(
        f"\nFace results saved to:"
        f"\n{FRAME_RESULTS_PATH}"
    )


def run_av_consistency(
    audio_path,
    frame_results
):

    print("\n" + "=" * 60)
    print("STEP 4 - AUDIO-VISUAL CONSISTENCY")
    print("=" * 60)

    if len(frame_results) < 2:

        print(
            "Not enough processed frames."
        )

        return

    visual_samples = sum(
        1
        for frame in frame_results
        if frame["faces"]
    )

    if visual_samples < 2:

        print(
            "Not enough face detections "
            "for AV consistency."
        )

        print(
            "The video may not contain a "
            "detectable face."
        )

        return

    print(
        f"Frames available for AV analysis: "
        f"{len(frame_results)}"
    )

    print(
        f"Frames containing faces: "
        f"{visual_samples}"
    )

    result = calculate_av_consistency(
        audio_path=audio_path,
        frame_results=frame_results,
        sample_rate=16000,
        hop_length=512
    )

    print("\n" + "=" * 60)
    print("AV CONSISTENCY RESULT")
    print("=" * 60)

    print(
        f"Status: "
        f"{result.get('status')}"
    )

    print(
        f"Audio samples: "
        f"{result.get('audio_samples')}"
    )

    print(
        f"Visual samples: "
        f"{result.get('visual_samples')}"
    )

    print(
        f"Audio duration: "
        f"{result.get('audio_duration_seconds')} sec"
    )

    print(
        f"Correlation: "
        f"{result.get('correlation')}"
    )

    print(
        f"Consistency score: "
        f"{result.get('consistency_score')}"
    )

    # Save complete AV result

    av_output_path = os.path.join(
        OUTPUT_DIR,
        "av_consistency.json"
    )

    with open(
        av_output_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        f"\nAV result saved to:"
        f"\n{av_output_path}"
    )


def main():

    print("=" * 60)
    print("DeepScan - Complete AV Consistency Test")
    print("=" * 60)

    # --------------------------------------------------
    # Check video
    # --------------------------------------------------

    if not os.path.exists(
        VIDEO_PATH
    ):

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{VIDEO_PATH}"
        )

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    audio_path = extract_audio()

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    frames = extract_sampled_frames()

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    frame_results = run_retinaface(
        frames
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    save_face_results(
        frame_results
    )

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    run_av_consistency(
        audio_path,
        frame_results
    )

    print("\n" + "=" * 60)
    print("COMPLETE AV TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()