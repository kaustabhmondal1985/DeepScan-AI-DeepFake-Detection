from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\Kaustabh\Downloads\FaceForensics++_test"
)

OUTPUT_PATH = Path(
    "training/dataset_manifest.csv"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

MANIPULATION_TYPES = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

REAL_YOUTUBE_DIR = (
    DATASET_ROOT
    / "original_sequences"
    / "youtube"
    / "c23"
    / "videos"
)

MANIPULATED_ROOT = (
    DATASET_ROOT
    / "manipulated_sequences"
)


# ============================================================
# SOURCE PAIRS
# ============================================================

SOURCE_PAIRS = [
    ("033", "097"),
    ("035", "036"),
    ("044", "945"),
    ("134", "192"),
    ("183", "253"),
    ("210", "241"),
    ("252", "266"),
    ("339", "392"),
    ("469", "481"),
    ("585", "599"),
    ("672", "720"),
    ("828", "830"),
    ("866", "878"),
    ("917", "924"),
    ("942", "943"),
]


# Create lookup:
#
# 033 -> 033_097
# 097 -> 033_097
# 035 -> 035_036
# 036 -> 035_036
# etc.

VIDEO_TO_SOURCE_GROUP = {}

for first, second in SOURCE_PAIRS:

    group = f"{first}_{second}"

    VIDEO_TO_SOURCE_GROUP[first] = group
    VIDEO_TO_SOURCE_GROUP[second] = group


# ============================================================
# SOURCE GROUP FUNCTION
# ============================================================

def get_source_group(filename: str) -> str | None:

    stem = Path(filename).stem

    # --------------------------------------------------------
    # Real video
    #
    # Example:
    # 033.mp4 -> 033_097
    # --------------------------------------------------------

    if "_" not in stem:

        if stem in VIDEO_TO_SOURCE_GROUP:
            return VIDEO_TO_SOURCE_GROUP[stem]

        return None

    # --------------------------------------------------------
    # Manipulated video
    #
    # Example:
    # 033_097.mp4 -> 033_097
    # 097_033.mp4 -> 033_097
    # --------------------------------------------------------

    parts = stem.split("_")

    if len(parts) != 2:
        return None

    first = parts[0].zfill(3)
    second = parts[1].zfill(3)

    pair = tuple(
        sorted([first, second])
    )

    group = f"{pair[0]}_{pair[1]}"

    if group in {
        f"{a}_{b}"
        for a, b in SOURCE_PAIRS
    }:
        return group

    return None


# ============================================================
# BUILD MANIFEST
# ============================================================

def build_manifest():

    records = []

    print("=" * 70)
    print("DeepScan - Building Dataset Manifest")
    print("=" * 70)

    # --------------------------------------------------------
    # REAL YOUTUBE VIDEOS
    # --------------------------------------------------------

    real_files = sorted(
        REAL_YOUTUBE_DIR.glob("*.mp4")
    )

    print(
        f"\nReal YouTube videos found: "
        f"{len(real_files)}"
    )

    for video_path in real_files:

        source_group = get_source_group(
            video_path.name
        )

        if source_group is None:

            print(
                f"[WARNING] Unknown real source: "
                f"{video_path.name}"
            )

            continue

        records.append({
            "video_path": str(video_path),
            "label": 0,
            "label_name": "real",
            "manipulation": "original",
            "source_group": source_group,
            "video_name": video_path.name,
        })

    # --------------------------------------------------------
    # MANIPULATED VIDEOS
    # --------------------------------------------------------

    for manipulation in MANIPULATION_TYPES:

        manipulation_dir = (
            MANIPULATED_ROOT
            / manipulation
            / "c23"
            / "videos"
        )

        files = sorted(
            manipulation_dir.glob("*.mp4")
        )

        print(
            f"{manipulation:15s}: "
            f"{len(files)} videos"
        )

        for video_path in files:

            source_group = get_source_group(
                video_path.name
            )

            if source_group is None:

                print(
                    f"[WARNING] Unknown manipulated source: "
                    f"{video_path.name}"
                )

                continue

            records.append({
                "video_path": str(video_path),
                "label": 1,
                "label_name": "fake",
                "manipulation": manipulation,
                "source_group": source_group,
                "video_name": video_path.name,
            })

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            "No dataset records were found."
        )

    df = df.sort_values(
        by=[
            "source_group",
            "label",
            "manipulation",
            "video_name",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MANIFEST SUMMARY")
    print("=" * 70)

    print(
        f"Total videos: {len(df)}"
    )

    print(
        f"Real videos: "
        f"{(df['label'] == 0).sum()}"
    )

    print(
        f"Fake videos: "
        f"{(df['label'] == 1).sum()}"
    )

    print(
        f"Source groups: "
        f"{df['source_group'].nunique()}"
    )

    print("\nManipulation breakdown:")

    manipulation_counts = (
        df[df["label"] == 1]
        ["manipulation"]
        .value_counts()
        .sort_index()
    )

    for name, count in manipulation_counts.items():

        print(
            f"  {name}: {count}"
        )

    print("\nSource groups:")

    group_counts = (
        df.groupby("source_group")
        .size()
        .sort_index()
    )

    for group, count in group_counts.items():

        print(
            f"  {group}: {count} videos"
        )

    print(
        f"\nManifest saved to: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_manifest()