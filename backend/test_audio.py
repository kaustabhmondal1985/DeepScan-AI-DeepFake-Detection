import os

from app.ml.audio.audio_processor import extract_audio


VIDEO_PATH = (
    r"C:\Users\Kaustabh\Downloads\FaceForensics++_test"
    r"\audio_test\sample_960x400_ocean_with_audio.mp4"
)

OUTPUT_DIR = "outputs/audio_test"


def main():

    print("=" * 60)
    print("DeepScan - Audio Extraction Test")
    print("=" * 60)

    print("\n[1] Checking input video...")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    print(
        f"[DeepScan] Video found:\n{VIDEO_PATH}"
    )

    print("\n[2] Running audio extraction...")

    result = extract_audio(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=16000
    )

    print("\n" + "=" * 60)
    print("AUDIO EXTRACTION RESULT")
    print("=" * 60)

    print(f"Status: {result['status']}")
    print(f"Reason: {result['reason']}")
    print(f"Audio path: {result['audio_path']}")
    print(f"Sample rate: {result['sample_rate']}")
    print(f"Channels: {result['channels']}")
    print(f"Format: {result['format']}")
    print(f"Codec: {result['codec']}")
    print(
        f"File size: "
        f"{result['file_size_bytes']} bytes"
    )

    print("\n" + "=" * 60)
    print("AUDIO PIPELINE VERIFICATION")
    print("=" * 60)

    if result["status"] == "unavailable":

        print("Audio stream available: NO")
        print("Audio extraction: SKIPPED")
        print("Pipeline behavior: CORRECT")

        print(
            "\nThis video can continue through "
            "the visual/temporal pipeline."
        )

    elif result["status"] == "success":

        audio_path = result["audio_path"]

        if os.path.exists(audio_path):

            print("Audio stream available: YES")
            print("Audio file exists: YES")
            print("Audio extraction: SUCCESS")

        else:

            print("Audio stream available: YES")
            print("Audio file exists: NO")
            print("Audio extraction: FAILED")

    else:

        print("Unexpected audio status.")

    print("\n" + "=" * 60)
    print("AUDIO EXTRACTION TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()