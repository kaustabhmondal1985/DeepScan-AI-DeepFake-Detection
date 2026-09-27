from typing import Dict, Any, List

import torch

from app.ml.temporal.lstm_model import TemporalLSTM


class TemporalAnalyzer:

    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 256,
        num_layers: int = 2
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[DeepScan] Temporal LSTM device: "
            f"{self.device}"
        )

        self.model = TemporalLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "[DeepScan] Temporal LSTM ready."
        )

    @torch.no_grad()
    def analyze_track(
        self,
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:

        if not embeddings:
            return {
                "sequence_length": 0,
                "embedding_dimension": 0,
                "temporal_embedding_dimension": 0,
                "temporal_embedding": []
            }

        sequence = torch.tensor(
            embeddings,
            dtype=torch.float32
        )

        # [sequence_length, 2048]
        sequence_length = sequence.shape[0]
        embedding_dimension = sequence.shape[1]

        # Add batch dimension
        # [1, sequence_length, 2048]
        sequence = sequence.unsqueeze(0)

        sequence = sequence.to(
            self.device
        )

        _, temporal_embedding = (
            self.model(sequence)
        )

        temporal_embedding = (
            temporal_embedding
            .squeeze(0)
            .cpu()
        )

        return {
            "sequence_length": sequence_length,
            "embedding_dimension": embedding_dimension,
            "temporal_embedding_dimension": (
                temporal_embedding.shape[0]
            ),
            "temporal_embedding": (
                temporal_embedding
                .numpy()
                .tolist()
            )
        }

    def analyze_tracks(
        self,
        tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        results = []

        print(
            f"[DeepScan] Analyzing "
            f"{len(tracks)} face tracks "
            f"temporally..."
        )

        for track in tracks:

            track_id = track["track_id"]

            print(
                f"[DeepScan] LSTM processing "
                f"track {track_id}..."
            )

            result = self.analyze_track(
                track["embeddings"]
            )

            result["track_id"] = track_id

            results.append(result)

            print(
                f"[DeepScan] Track {track_id} "
                f"temporal analysis complete."
            )

        return {
            "tracks_analyzed": len(results),
            "temporal_embedding_dimension": (
                256
                if results
                else 0
            ),
            "tracks": results
        }