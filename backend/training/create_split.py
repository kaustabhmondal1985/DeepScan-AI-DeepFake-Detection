from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = Path(
    "training/dataset_manifest.csv"
)

OUTPUT_DIR = Path(
    "training/splits"
)


# ============================================================
# SOURCE GROUP SPLIT
# ============================================================

# 15 total source groups.
#
# 9 groups  -> TRAIN
# 3 groups  -> VALIDATION
# 3 groups  -> TEST
#
# IMPORTANT:
# A complete source group stays in exactly one split.

TRAIN_GROUPS = [
    "033_097",
    "035_036",
    "044_945",
    "134_192",
    "183_253",
    "210_241",
    "252_266",
    "339_392",
    "469_481",
]

VAL_GROUPS = [
    "585_599",
    "672_720",
    "828_830",
]

TEST_GROUPS = [
    "866_878",
    "917_924",
    "942_943",
]


# ============================================================
# VALIDATION
# ============================================================

def validate_groups():

    train_set = set(TRAIN_GROUPS)
    val_set = set(VAL_GROUPS)
    test_set = set(TEST_GROUPS)

    # Check duplicate groups
    if len(train_set) != len(TRAIN_GROUPS):
        raise ValueError(
            "Duplicate source group found in TRAIN."
        )

    if len(val_set) != len(VAL_GROUPS):
        raise ValueError(
            "Duplicate source group found in VALIDATION."
        )

    if len(test_set) != len(TEST_GROUPS):
        raise ValueError(
            "Duplicate source group found in TEST."
        )

    # Check leakage
    if train_set & val_set:
        raise ValueError(
            "Source-group leakage between TRAIN and VALIDATION."
        )

    if train_set & test_set:
        raise ValueError(
            "Source-group leakage between TRAIN and TEST."
        )

    if val_set & test_set:
        raise ValueError(
            "Source-group leakage between VALIDATION and TEST."
        )

    all_groups = (
        train_set |
        val_set |
        test_set
    )

    if len(all_groups) != 15:
        raise ValueError(
            f"Expected 15 source groups, "
            f"but found {len(all_groups)}."
        )

    print(
        "[DeepScan] Source-group leakage check: PASSED"
    )


# ============================================================
# CREATE SPLITS
# ============================================================

def main():

    print("=" * 70)
    print("DeepScan - Creating Leakage-Safe Dataset Split")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    validate_groups()

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"\nTotal manifest records: {len(df)}"
    )

    # --------------------------------------------------------
    # ASSIGN SPLIT
    # --------------------------------------------------------

    def assign_split(source_group):

        if source_group in TRAIN_GROUPS:
            return "train"

        if source_group in VAL_GROUPS:
            return "val"

        if source_group in TEST_GROUPS:
            return "test"

        raise ValueError(
            f"Unknown source group: {source_group}"
        )

    df["split"] = (
        df["source_group"]
        .apply(assign_split)
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "val.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY FUNCTION
    # --------------------------------------------------------

    def print_summary(
        name,
        split_df,
        groups
    ):

        real_count = (
            split_df["label"] == 0
        ).sum()

        fake_count = (
            split_df["label"] == 1
        ).sum()

        print(
            f"\n{name}"
        )

        print(
            f"  Videos: {len(split_df)}"
        )

        print(
            f"  Real: {real_count}"
        )

        print(
            f"  Fake: {fake_count}"
        )

        print(
            f"  Source groups: "
            f"{', '.join(groups)}"
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print_summary(
        "TRAIN",
        train_df,
        TRAIN_GROUPS
    )

    print_summary(
        "VALIDATION",
        val_df,
        VAL_GROUPS
    )

    print_summary(
        "TEST",
        test_df,
        TEST_GROUPS
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Split files saved:"
    )

    print(
        f"  {train_path}"
    )

    print(
        f"  {val_path}"
    )

    print(
        f"  {test_path}"
    )

    print(
        "\nSource-group leakage: NONE"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()