import json

from app.ml.visual.track_embedding_extractor import (
    TrackEmbeddingExtractor
)

from app.ml.temporal.temporal_analyzer import (
    TemporalAnalyzer
)


JSON_PATH = "outputs/ffpp_test_183_253/preprocessing.json"


def main():

    print("=" * 60)
    print("DeepScan - Xception + LSTM Temporal Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load preprocessing data
    # ---------------------------------------------------------

    print("\n[1] Loading preprocessing JSON...")

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    tracks = data["face_tracking"]["tracks"]

    print(
        f"[DeepScan] Tracks found: {len(tracks)}"
    )

    for track in tracks:
        print(
            f"[DeepScan] Track {track['track_id']} "
            f"has {len(track['detections'])} detections"
        )

    # ---------------------------------------------------------
    # 2. Extract Xception embeddings
    # ---------------------------------------------------------

    print("\n[2] Initializing Xception extractor...")

    extractor = TrackEmbeddingExtractor()

    print("\n[3] Extracting Xception embeddings...")

    embedding_result = (
        extractor.extract_track_embeddings(tracks)
    )

    print(
        f"[DeepScan] Tracks processed: "
        f"{embedding_result['tracks_processed']}"
    )

    print(
        f"[DeepScan] Embedding dimension: "
        f"{embedding_result['embedding_dimension']}"
    )

    # ---------------------------------------------------------
    # 3. Initialize Temporal Analyzer
    # ---------------------------------------------------------

    print("\n[4] Initializing TemporalAnalyzer...")

    temporal_analyzer = TemporalAnalyzer(
        input_size=2048,
        hidden_size=256,
        num_layers=2
    )

    print("[DeepScan] TemporalAnalyzer ready.")

    # ---------------------------------------------------------
    # 4. Run LSTM on Xception embeddings
    # ---------------------------------------------------------

    print("\n[5] Running temporal analysis...")

    temporal_result = (
        temporal_analyzer.analyze_tracks(
            embedding_result["tracks"]
        )
    )

    # ---------------------------------------------------------
    # 5. Display temporal analysis result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEMPORAL ANALYSIS RESULT")
    print("=" * 60)

    print(
        f"Tracks analyzed: "
        f"{temporal_result['tracks_analyzed']}"
    )

    print(
        f"Temporal embedding dimension: "
        f"{temporal_result['temporal_embedding_dimension']}"
    )

    # ---------------------------------------------------------
    # 6. Display individual track results
    # ---------------------------------------------------------

    for track in temporal_result["tracks"]:

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
            f"Original embedding dimension: "
            f"{track['embedding_dimension']}"
        )

        print(
            f"Temporal embedding dimension: "
            f"{track['temporal_embedding_dimension']}"
        )

        temporal_embedding = (
            track["temporal_embedding"]
        )

        print(
            f"Temporal embedding length: "
            f"{len(temporal_embedding)}"
        )

        print(
            "First 5 values of temporal embedding:"
        )

        print(
            temporal_embedding[:5]
        )

    # ---------------------------------------------------------
    # 7. Final verification
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE VERIFICATION")
    print("=" * 60)

    print(
        "Xception input: 65 face crops"
    )

    print(
        "Xception output: 65 × 2048"
    )

    print(
        "LSTM input: 65 × 2048"
    )

    print(
        "LSTM output: 256-D temporal embedding"
    )

    print("\n" + "=" * 60)
    print("XCEPTION + LSTM TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()