import os
import json

from app.ml.visual.track_embedding_extractor import (
    TrackEmbeddingExtractor
)

from app.ml.temporal.temporal_analyzer import (
    TemporalAnalyzer
)

from app.ml.audio.feature_extractor import (
    extract_mel_spectrogram
)

from app.ml.audio.audio_analyzer import (
    AudioAnalyzer
)

from app.ml.multimodal.av_consistency import (
    calculate_av_consistency
)

from app.ml.multimodal.fusion import (
    fuse_multimodal_evidence
)


# ============================================================
# Configuration
# ============================================================

ANALYSIS_ID = "DS-20260922-1CB46B"

PREPROCESSING_PATH = (
    rf"outputs\{ANALYSIS_ID}\preprocessing.json"
)

AUDIO_PATH = (
    r"outputs\audio_test\audio.wav"
)


# ============================================================
# File checks
# ============================================================

print("=" * 60)
print("DeepScan Multimodal Pipeline Test")
print("=" * 60)

if not os.path.exists(PREPROCESSING_PATH):
    raise FileNotFoundError(
        f"Preprocessing file not found: "
        f"{PREPROCESSING_PATH}"
    )

if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_PATH}"
    )

print("\n[Test] Required files found.")


# ============================================================
# Load preprocessing data
# ============================================================

print("\n[1/6] Loading preprocessing data...")

with open(
    PREPROCESSING_PATH,
    "r",
    encoding="utf-8"
) as file:
    preprocessing = json.load(file)

tracks = preprocessing[
    "face_tracking"
]["tracks"]

frame_results = preprocessing[
    "face_detection"
]["frames"]

print(
    f"[Test] Face tracks available: {len(tracks)}"
)

print(
    f"[Test] Frames available: {len(frame_results)}"
)


# ============================================================
# VISUAL BRANCH
# ============================================================

print("\n[2/6] Running visual branch...")

track_extractor = TrackEmbeddingExtractor()

track_embedding_result = (
    track_extractor.extract_track_embeddings(
        tracks
    )
)

print(
    "[Test] Tracks processed:",
    track_embedding_result[
        "tracks_processed"
    ]
)

print(
    "[Test] Visual embedding dimension:",
    track_embedding_result[
        "embedding_dimension"
    ]
)


# ============================================================
# TEMPORAL BRANCH
# ============================================================

print("\n[3/6] Running temporal branch...")

temporal_analyzer = TemporalAnalyzer(
    input_size=2048,
    hidden_size=256,
    num_layers=2
)

temporal_result = (
    temporal_analyzer.analyze_tracks(
        track_embedding_result["tracks"]
    )
)

print(
    "[Test] Tracks analyzed:",
    temporal_result[
        "tracks_analyzed"
    ]
)

print(
    "[Test] Temporal embedding dimension:",
    temporal_result[
        "temporal_embedding_dimension"
    ]
)


# ============================================================
# AUDIO BRANCH
# ============================================================

print("\n[4/6] Running audio branch...")

feature_result = extract_mel_spectrogram(
    audio_path=AUDIO_PATH,
    sample_rate=16000
)

mel_spectrogram = feature_result[
    "mel_spectrogram"
]

print(
    "[Test] Mel spectrogram shape:",
    mel_spectrogram.shape
)

audio_analyzer = AudioAnalyzer(
    embedding_size=256
)

audio_result = audio_analyzer.generate_embedding(
    mel_spectrogram=mel_spectrogram
)

print(
    "[Test] Audio embedding dimension:",
    audio_result[
        "embedding_dimension"
    ]
)


# ============================================================
# AUDIO-VISUAL CONSISTENCY
# ============================================================

print("\n[5/6] Running AV consistency...")

av_result = calculate_av_consistency(
    audio_path=AUDIO_PATH,
    frame_results=frame_results,
    sample_rate=16000,
    hop_length=512
)

print(
    "[Test] AV status:",
    av_result["status"]
)

print(
    "[Test] AV correlation:",
    av_result.get("correlation")
)

print(
    "[Test] AV consistency:",
    av_result.get(
        "consistency_score"
    )
)


# ============================================================
# MULTIMODAL FUSION
# ============================================================

print("\n[6/6] Running multimodal fusion...")


visual_evidence = {
    "status": "available",
    "tracks_processed": (
        track_embedding_result[
            "tracks_processed"
        ]
    ),
    "embedding_dimension": (
        temporal_result[
            "temporal_embedding_dimension"
        ]
    ),
    "tracks": temporal_result[
        "tracks"
    ]
}

audio_evidence = {
    "status": "available",
    "embedding_dimension": (
        audio_result[
            "embedding_dimension"
        ]
    ),
    "embedding": audio_result[
        "embedding"
    ]
}

fusion_result = fuse_multimodal_evidence(
    visual_result=visual_evidence,
    audio_result=audio_evidence,
    av_result=av_result
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL MULTIMODAL RESULT")
print("=" * 60)

print(
    "\nAssessment:",
    fusion_result["assessment"]
)

print(
    "Confidence:",
    fusion_result["confidence"]
)

print(
    "Evidence sources:",
    fusion_result["evidence_sources"]
)

print(
    "\nMessage:",
    fusion_result["message"]
)

print("\n[Test] DeepScan multimodal pipeline successful!")