import os
import json

from app.ml.preprocessing.frame_extractor import (
    extract_frames
)


VIDEO_PATH = (
    r"C:\Users\Kaustabh\Downloads\FaceForensics++_test"
    r"\audio_test\sample_960x400_ocean_with_audio.mp4"
)

OUTPUT_DIR = (
    "outputs/av_test"
)


def main():

    print("=" * 60)
    print("DeepScan - AV Consistency Preprocessing Test")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Check video
    # --------------------------------------------------

    print("\n[1] Checking input video...")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    print(
        f"[DeepScan] Video found:\n"
        f"{VIDEO_PATH}"
    )

    # --------------------------------------------------
    # 2. Extract frames at 5 FPS
    # --------------------------------------------------

    print("\n[2] Extracting frames at 5 FPS...")

    result = extract_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        sample_fps=5.0,
        max_frames=300
    )

    print("\n" + "=" * 60)
    print("FRAME EXTRACTION RESULT")
    print("=" * 60)

    print(
        f"Sample FPS: "
        f"{result['sample_fps']}"
    )

    print(
        f"Frames extracted: "
        f"{result['frames_extracted']}"
    )

    # --------------------------------------------------
    # 3. Save preprocessing result
    # --------------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        "frames.json"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        f"\n[DeepScan] Frame metadata saved:"
        f"\n{json_path}"
    )

    # --------------------------------------------------
    # 4. Verification
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FRAME PREPROCESSING VERIFICATION")
    print("=" * 60)

    if result["frames_extracted"] > 0:

        print(
            "Frames extracted: YES"
        )

        print(
            "Sample rate: 5 FPS"
        )

        print(
            "Frame extraction: SUCCESS"
        )

    else:

        print(
            "Frame extraction: FAILED"
        )

    print("\n" + "=" * 60)
    print("AV PREPROCESSING TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()