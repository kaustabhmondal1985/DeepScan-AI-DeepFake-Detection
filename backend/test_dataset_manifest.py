import os

from app.data.manifest import create_manifest


ORIGINAL_DIR = (
    r"..\datasets\faceforensics\original"
)

MANIPULATED_DIR = (
    r"..\datasets\faceforensics\manipulated"
)

OUTPUT_CSV = (
    r"..\datasets\metadata\manifest.csv"
)


print("=" * 60)
print("DeepScan Dataset Manifest Test")
print("=" * 60)


print("\n[Test] Checking dataset directories...")

if not os.path.exists(ORIGINAL_DIR):
    raise FileNotFoundError(
        f"Original directory not found: "
        f"{ORIGINAL_DIR}"
    )

if not os.path.exists(MANIPULATED_DIR):
    raise FileNotFoundError(
        f"Manipulated directory not found: "
        f"{MANIPULATED_DIR}"
    )


records = create_manifest(
    original_dir=ORIGINAL_DIR,
    manipulated_dir=MANIPULATED_DIR,
    output_csv=OUTPUT_CSV
)


real_count = sum(
    1 for record in records
    if record["label"] == 0
)

fake_count = sum(
    1 for record in records
    if record["label"] == 1
)


print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(
    f"Total videos: {len(records)}"
)

print(
    f"Real videos: {real_count}"
)

print(
    f"Manipulated videos: {fake_count}"
)

print(
    f"Manifest: {OUTPUT_CSV}"
)

print(
    "\n[Test] Dataset manifest generation successful!"
)