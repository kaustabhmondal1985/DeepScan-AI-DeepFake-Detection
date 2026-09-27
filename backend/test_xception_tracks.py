import json

from app.ml.visual.track_embedding_extractor import (
    TrackEmbeddingExtractor
)


JSON_PATH = "outputs/ffpp_test_183_253/preprocessing.json"


def main():

    print("=" * 60)
    print("DeepScan - Xception Track Embedding Test")
    print("=" * 60)

    # Load preprocessing result
    print("\n[1] Loading preprocessing JSON...")

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Get tracked faces
    tracks = data["face_tracking"]["tracks"]

    print(f"[DeepScan] Tracks found: {len(tracks)}")

    for track in tracks:
        print(
            f"[DeepScan] Track {track['track_id']} "
            f"has {len(track['detections'])} detections"
        )

    # Initialize Xception-based extractor
    print("\n[2] Initializing TrackEmbeddingExtractor...")

    extractor = TrackEmbeddingExtractor()

    # Extract embeddings
    print("\n[3] Extracting Xception embeddings...")

    result = extractor.extract_track_embeddings(tracks)

    # Display results
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"Tracks processed: "
        f"{result['tracks_processed']}"
    )

    print(
        f"Embedding dimension: "
        f"{result['embedding_dimension']}"
    )

    for track in result["tracks"]:

        print("\n----------------------------------------")

        print(
            f"Track ID: "
            f"{track['track_id']}"
        )

        print(
            f"Sequence length: "
            f"{track['sequence_length']}"
        )

        print(
            f"Embedding dimension: "
            f"{track['embedding_dimension']}"
        )

        print(
            f"Number of timestamps: "
            f"{len(track['timestamps'])}"
        )

        print(
            f"Number of crop paths: "
            f"{len(track['crop_paths'])}"
        )

        # Check first embedding
        first_embedding = track["embeddings"][0]

        print(
            f"First embedding length: "
            f"{len(first_embedding)}"
        )

        print(
            "First 5 values of first embedding:"
        )

        print(first_embedding[:5])

    print("\n" + "=" * 60)
    print("XCEPTION EMBEDDING TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()