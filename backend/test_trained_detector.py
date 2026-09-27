import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.analysis_pipeline import DeepScanPipeline


def main():

    print("=" * 70)
    print("DeepScan - Trained Detector Test")
    print("=" * 70)

    pipeline = DeepScanPipeline()

    preprocessing_file = (
        BACKEND_DIR
        / "outputs"
        / "ffpp_test_183_253"
        / "preprocessing.json"
    )

    print("\n[DeepScan] Input:")
    print(preprocessing_file)

    if not preprocessing_file.exists():
        raise FileNotFoundError(
            f"Preprocessing file not found:\n"
            f"{preprocessing_file}"
        )

    result = pipeline.analyze(
        str(preprocessing_file)
    )

    print("\n" + "=" * 70)
    print("DETECTION RESULT")
    print("=" * 70)

    print(
        f"Assessment : "
        f"{result['assessment']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence']}"
    )

    print(
        f"Tracks     : "
        f"{result['tracks_analyzed']}"
    )

    print(
        f"Real       : "
        f"{result['probabilities']['real']}"
    )

    print(
        f"Fake       : "
        f"{result['probabilities']['fake']}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()