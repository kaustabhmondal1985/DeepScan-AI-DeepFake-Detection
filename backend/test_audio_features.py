import os
import numpy as np

from app.ml.audio.feature_extractor import (
    extract_mel_spectrogram
)


AUDIO_PATH = (
    "outputs/audio_test/audio.wav"
)


def main():

    print("=" * 60)
    print("DeepScan - Audio Feature Extraction Test")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Check extracted audio
    # --------------------------------------------------

    print("\n[1] Checking extracted audio...")

    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_PATH}"
        )

    print(
        f"[DeepScan] Audio found:\n"
        f"{AUDIO_PATH}"
    )

    # --------------------------------------------------
    # 2. Extract Mel Spectrogram
    # --------------------------------------------------

    print("\n[2] Extracting Mel Spectrogram...")

    result = extract_mel_spectrogram(
        audio_path=AUDIO_PATH,
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )

    # --------------------------------------------------
    # 3. Display result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIO FEATURE EXTRACTION RESULT")
    print("=" * 60)

    print(
        f"Sample rate: "
        f"{result['sample_rate']} Hz"
    )

    print(
        f"Duration: "
        f"{result['duration_seconds']} seconds"
    )

    print(
        f"N_FFT: "
        f"{result['n_fft']}"
    )

    print(
        f"Hop length: "
        f"{result['hop_length']}"
    )

    print(
        f"Mel bands: "
        f"{result['n_mels']}"
    )

    print(
        f"Mel shape: "
        f"{result['mel_shape']}"
    )

    print(
        f"Mel mean: "
        f"{result['mel_mean']:.6f}"
    )

    print(
        f"Mel standard deviation: "
        f"{result['mel_std']:.6f}"
    )

    # --------------------------------------------------
    # 4. Inspect actual spectrogram
    # --------------------------------------------------

    mel = result["mel_spectrogram"]

    print(
        f"Actual NumPy shape: "
        f"{mel.shape}"
    )

    print(
        f"Data type: "
        f"{mel.dtype}"
    )

    print(
        "First 5 values from first Mel band:"
    )

    print(
        mel[0, :5]
    )

    # --------------------------------------------------
    # 5. Verification
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIO FEATURE VERIFICATION")
    print("=" * 60)

    checks = {

        "Sample rate is 16000 Hz":
            result["sample_rate"] == 16000,

        "Mel bands are 128":
            result["n_mels"] == 128,

        "N_FFT is 1024":
            result["n_fft"] == 1024,

        "Hop length is 512":
            result["hop_length"] == 512,

        "Spectrogram has 128 Mel bands":
            mel.shape[0] == 128,

        "Time dimension exists":
            mel.shape[1] > 0,

        "Output is float32":
            mel.dtype == np.float32,

        "Spectrogram contains values":
            mel.size > 0

    }

    all_passed = True

    for check_name, passed in checks.items():

        if passed:
            print(
                f"[PASS] {check_name}"
            )
        else:
            print(
                f"[FAIL] {check_name}"
            )
            all_passed = False

    # --------------------------------------------------
    # 6. Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if all_passed:

        print(
            "Mel spectrogram extraction: SUCCESS"
        )

        print(
            f"Final feature shape: "
            f"{mel.shape[0]} × {mel.shape[1]}"
        )

        print(
            "Audio feature pipeline: VERIFIED"
        )

    else:

        print(
            "Mel spectrogram extraction: FAILED"
        )

    print("\n" + "=" * 60)
    print("AUDIO FEATURE TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()