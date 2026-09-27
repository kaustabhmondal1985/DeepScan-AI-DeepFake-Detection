import json

from app.ml.visual.track_embedding_extractor import (
    TrackEmbeddingExtractor
)


PREPROCESSING_PATH = (
    r"outputs\DS-20260922-1CB46B"
    r"\preprocessing.json"
)


print("[Test] Loading preprocessing data...")

with open(
    PREPROCESSING_PATH,
    "r",
    encoding="utf-8"
) as file:
    preprocessing = json.load(file)


tracks = preprocessing[
    "face_tracking"
]["tracks"]


print(
    f"[Test] Found {len(tracks)} tracks."
)

extractor = TrackEmbeddingExtractor()

result = extractor.extract_track_embeddings(
    tracks
)

print("\n[Test] Result summary:")
print(
    "Tracks processed:",
    result["tracks_processed"]
)

print(
    "Embedding dimension:",
    result["embedding_dimension"]
)

for track in result["tracks"]:

    print(
        f"Track {track['track_id']}: "
        f"{track['sequence_length']} frames → "
        f"{track['embedding_dimension']}-D"
    )

print(
    "\n[Test] Track embedding extraction successful!"
)