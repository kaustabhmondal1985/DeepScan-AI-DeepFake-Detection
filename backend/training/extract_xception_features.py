from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from app.ml.visual.xception_model import XceptionFeatureExtractor


# ============================================================
# PATHS
# ============================================================

SPLIT_DIR = Path("training/splits")

PROCESSED_DIR = Path(
    "training/processed"
)

FEATURE_DIR = Path(
    "training/features"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16


# ============================================================
# XCEPTION INITIALIZATION
# ============================================================

print("=" * 70)
print("DeepScan - XCEPTION FEATURE EXTRACTION")
print("=" * 70)

print()
print("[DeepScan] Initializing Xception...")

extractor = XceptionFeatureExtractor()

print("[DeepScan] Xception ready.")
print()


# ============================================================
# LOAD VIDEO METADATA
# ============================================================

def load_video_metadata(
    split_name,
    row
):
    """
    Load the preprocessing metadata for one video.
    """

    video_name = Path(
        row["video_path"]
    ).stem

    manipulation = row[
        "manipulation"
    ]

    metadata_path = (
        PROCESSED_DIR
        / split_name
        / f"{manipulation}_{video_name}"
        / "metadata.json"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata not found: "
            f"{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# OUTPUT PATH
# ============================================================

def get_feature_path(
    split_name,
    row
):

    video_name = Path(
        row["video_path"]
    ).stem

    manipulation = row[
        "manipulation"
    ]

    output_dir = (
        FEATURE_DIR
        / split_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        output_dir
        / f"{manipulation}_{video_name}.npz"
    )


# ============================================================
# EXTRACT FEATURES FOR ONE VIDEO
# ============================================================

def extract_video_features(
    split_name,
    row
):

    metadata = load_video_metadata(
        split_name,
        row
    )

    feature_path = get_feature_path(
        split_name,
        row
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if feature_path.exists():

        try:

            existing = np.load(
                feature_path
            )

            embeddings = existing[
                "embeddings"
            ]

            if embeddings.shape[0] > 0:

                print(
                    "      [SKIP] Features "
                    "already extracted"
                )

                return {
                    "status": "success",
                    "skipped": True,
                    "embedding_shape": list(
                        embeddings.shape
                    )
                }

        except Exception:

            print(
                "      [REPROCESS] "
                "Existing feature file "
                "could not be loaded."
            )

    # --------------------------------------------------------
    # GET CROP PATHS
    # --------------------------------------------------------

    frames = metadata.get(
        "frames",
        []
    )

    valid_frames = []

    crop_paths = []

    timestamps = []

    frame_indices = []

    for frame in frames:

        if not frame.get(
            "face_detected",
            False
        ):
            continue

        crop_path = frame.get(
            "crop_path"
        )

        if not crop_path:
            continue

        crop_path = Path(
            crop_path
        )

        if not crop_path.exists():

            print(
                f"      [WARNING] "
                f"Missing crop: "
                f"{crop_path}"
            )

            continue

        valid_frames.append(
            frame
        )

        crop_paths.append(
            str(crop_path)
        )

        timestamps.append(
            float(
                frame.get(
                    "timestamp",
                    0.0
                )
            )
        )

        frame_indices.append(
            int(
                frame.get(
                    "frame_index",
                    0
                )
            )
        )

    if not crop_paths:

        raise RuntimeError(
            "No valid face crops found."
        )

    # --------------------------------------------------------
    # XCEPTION
    # --------------------------------------------------------

    print(
        f"      Extracting Xception "
        f"features from "
        f"{len(crop_paths)} crops..."
    )

    start = time.time()

    embeddings = extractor.extract_embeddings(
        crop_paths
    )

    elapsed = (
        time.time()
        - start
    )

    embeddings_np = (
        embeddings
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    np.savez_compressed(
        feature_path,
        embeddings=embeddings_np,
        timestamps=np.asarray(
            timestamps,
            dtype=np.float32
        ),
        frame_indices=np.asarray(
            frame_indices,
            dtype=np.int32
        ),
        label=np.asarray(
            int(row["label"]),
            dtype=np.int64
        )
    )

    print(
        f"      Saved: "
        f"{feature_path}"
    )

    print(
        f"      Shape: "
        f"{embeddings_np.shape}"
    )

    print(
        f"      Time: "
        f"{elapsed:.2f} sec"
    )

    return {
        "status": "success",
        "skipped": False,
        "embedding_shape": list(
            embeddings_np.shape
        )
    }


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split_name
):

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

    print()
    print("=" * 70)
    print(
        f"PROCESSING {split_name.upper()} "
        f"XCEPTION FEATURES"
    )
    print("=" * 70)

    split_start = time.time()

    successful = 0
    failed = 0
    skipped = 0

    total_embeddings = 0

    for index, row in df.iterrows():

        print()
        print(
            f"[{index + 1}/{len(df)}] "
            f"{row['video_name']} "
            f"| {row['label_name']} "
            f"| {row['manipulation']}"
        )

        try:

            result = extract_video_features(
                split_name,
                row
            )

            if result["status"] == "success":

                successful += 1

                if result.get(
                    "skipped",
                    False
                ):

                    skipped += 1

                shape = result.get(
                    "embedding_shape"
                )

                if shape:
                    total_embeddings += shape[0]

        except Exception as exc:

            failed += 1

            print(
                f"      [ERROR] "
                f"{exc}"
            )

    elapsed = (
        time.time()
        - split_start
    )

    print()
    print("=" * 70)
    print(
        f"{split_name.upper()} XCEPTION "
        f"FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Videos: "
        f"{len(df)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Skipped: "
        f"{skipped}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Embeddings: "
        f"{total_embeddings}"
    )

    print(
        f"Time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # Process all three splits.
    #
    # TRAIN:
    # 108 videos × 8 frames
    #
    # VAL:
    # 36 videos × 8 frames
    #
    # TEST:
    # 36 videos × 8 frames

    process_split("train")

    process_split("val")

    process_split("test")

    print()
    print("=" * 70)
    print(
        "ALL XCEPTION FEATURES EXTRACTED"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()