import os

from app.ml.visual.xception_model import (
    XceptionFeatureExtractor
)


CROP_PATH = (
    r"outputs\DS-20260922-39B74C"
    r"\face_crops\frame_000000_face_00.jpg"
)


print("[Test] Checking crop...")

if not os.path.exists(CROP_PATH):
    raise FileNotFoundError(
        f"Crop not found: {CROP_PATH}"
    )

print(
    "[Test] Crop found:",
    CROP_PATH
)

print(
    "[Test] Loading Xception..."
)

extractor = XceptionFeatureExtractor()

print(
    "[Test] Extracting embedding..."
)

embedding = extractor.extract_embedding(
    CROP_PATH
)

print(
    "[Test] Embedding shape:",
    embedding.shape
)

print(
    "[Test] Embedding dimension:",
    embedding.shape[0]
)

print(
    "[Test] First 10 values:"
)

print(
    embedding[:10]
)

print(
    "[Test] Xception feature extraction successful!"
)