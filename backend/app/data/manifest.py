import csv
from pathlib import Path
from typing import List, Dict


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


def discover_videos(
    root_dir: str,
    label: int,
    source_dataset: str,
    manipulation_type: str
) -> List[Dict]:

    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}"
        )

    records = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        records.append({
            "video_path": str(path.resolve()),
            "label": label,
            "source_dataset": source_dataset,
            "manipulation_type": manipulation_type
        })

    return records


def create_manifest(
    original_dir: str,
    manipulated_dir: str,
    output_csv: str
):

    records = []

    print("[DeepScan] Searching original videos...")

    original_records = discover_videos(
        root_dir=original_dir,
        label=0,
        source_dataset="FaceForensics++",
        manipulation_type="original"
    )

    records.extend(original_records)

    print(
        f"[DeepScan] Original videos found: "
        f"{len(original_records)}"
    )

    print("[DeepScan] Searching manipulated videos...")

    manipulated_records = discover_videos(
        root_dir=manipulated_dir,
        label=1,
        source_dataset="FaceForensics++",
        manipulation_type="manipulated"
    )

    records.extend(manipulated_records)

    print(
        f"[DeepScan] Manipulated videos found: "
        f"{len(manipulated_records)}"
    )

    output_path = Path(output_csv)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "video_path",
        "label",
        "source_dataset",
        "manipulation_type"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    print(
        f"[DeepScan] Manifest created: "
        f"{output_path}"
    )

    print(
        f"[DeepScan] Total videos: "
        f"{len(records)}"
    )

    return records