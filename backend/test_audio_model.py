import os

import torch

from app.ml.audio.feature_extractor import (
    extract_mel_spectrogram
)

from app.ml.audio.audio_model import (
    AudioCNN
)


AUDIO_PATH = (
    "outputs/audio_test/audio.wav"
)


def main():

    print("=" * 60)
    print("DeepScan - Audio CNN Test")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Check audio
    # --------------------------------------------------

    print("\n[1] Checking audio file...")

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

    feature_result = extract_mel_spectrogram(
        audio_path=AUDIO_PATH,
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )

    mel = feature_result[
        "mel_spectrogram"
    ]

    print(
        f"[DeepScan] Mel shape: "
        f"{mel.shape}"
    )

    # --------------------------------------------------
    # 3. Convert NumPy → PyTorch
    # --------------------------------------------------

    print(
        "\n[3] Preparing tensor for Audio CNN..."
    )

    # Current shape:
    # [128, time]

    audio_tensor = torch.from_numpy(
        mel
    )

    # Add channel dimension:
    # [1, 128, time]

    audio_tensor = audio_tensor.unsqueeze(
        0
    )

    # Add batch dimension:
    # [1, 1, 128, time]

    audio_tensor = audio_tensor.unsqueeze(
        0
    )

    audio_tensor = audio_tensor.float()

    print(
        f"[DeepScan] CNN input shape: "
        f"{tuple(audio_tensor.shape)}"
    )

    print(
        f"[DeepScan] Tensor dtype: "
        f"{audio_tensor.dtype}"
    )

    # --------------------------------------------------
    # 4. Initialize Audio CNN
    # --------------------------------------------------

    print(
        "\n[4] Initializing AudioCNN..."
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[DeepScan] Device: {device}"
    )

    model = AudioCNN(
        embedding_size=256
    )

    model = model.to(device)

    model.eval()

    print(
        "[DeepScan] AudioCNN ready."
    )

    # --------------------------------------------------
    # 5. Move input to device
    # --------------------------------------------------

    audio_tensor = audio_tensor.to(
        device
    )

    # --------------------------------------------------
    # 6. Run Audio CNN
    # --------------------------------------------------

    print(
        "\n[5] Running Audio CNN..."
    )

    with torch.no_grad():

        embedding = model(
            audio_tensor
        )

    # --------------------------------------------------
    # 7. Display result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIO CNN RESULT")
    print("=" * 60)

    print(
        f"Input shape: "
        f"{tuple(audio_tensor.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(embedding.shape)}"
    )

    print(
        f"Embedding dimension: "
        f"{embedding.shape[1]}"
    )

    print(
        f"Embedding dtype: "
        f"{embedding.dtype}"
    )

    embedding_cpu = (
        embedding
        .cpu()
        .numpy()
    )

    print(
        "\nFirst 10 embedding values:"
    )

    print(
        embedding_cpu[0][:10]
    )

    # --------------------------------------------------
    # 8. Verification
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIO CNN VERIFICATION")
    print("=" * 60)

    checks = {

        "Input has batch dimension":
            audio_tensor.shape[0] == 1,

        "Input has one channel":
            audio_tensor.shape[1] == 1,

        "Input has 128 Mel bands":
            audio_tensor.shape[2] == 128,

        "Input has time dimension":
            audio_tensor.shape[3] > 0,

        "Output batch size is 1":
            embedding.shape[0] == 1,

        "Output embedding is 256-D":
            embedding.shape[1] == 256,

        "Output contains finite values":
            torch.isfinite(
                embedding
            ).all().item(),

        "Output contains values":
            embedding.numel() == 256

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
    # 9. Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if all_passed:

        print(
            "Audio CNN inference: SUCCESS"
        )

        print(
            "Input: 1 × 1 × 128 × "
            f"{audio_tensor.shape[3]}"
        )

        print(
            "Output: 1 × 256"
        )

        print(
            "Audio embedding pipeline: VERIFIED"
        )

    else:

        print(
            "Audio CNN inference: FAILED"
        )

    print("\n" + "=" * 60)
    print("AUDIO CNN TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()