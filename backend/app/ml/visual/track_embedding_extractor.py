from typing import Dict, Any, List

import torch

from app.ml.visual.xception_model import (
    XceptionFeatureExtractor
)


class TrackEmbeddingExtractor:

    def __init__(self):
        print(
            "[DeepScan] Initializing "
            "TrackEmbeddingExtractor..."
        )

        self.extractor = XceptionFeatureExtractor()

        print(
            "[DeepScan] TrackEmbeddingExtractor ready."
        )

    def extract_track_embeddings(
        self,
        tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        track_results = []

        print(
            f"[DeepScan] Processing "
            f"{len(tracks)} face tracks..."
        )

        for track in tracks:

            track_id = track["track_id"]
            detections = track["detections"]

            print(
                f"[DeepScan] Processing track "
                f"{track_id} with "
                f"{len(detections)} detections..."
            )

            crop_paths = []

            for detection in detections:

                crop_path = detection.get(
                    "crop_path"
                )

                if crop_path:
                    crop_paths.append(
                        crop_path
                    )

            if not crop_paths:
                print(
                    f"[DeepScan] Track {track_id} "
                    f"has no valid crops."
                )
                continue

            embeddings = (
                self.extractor.extract_embeddings(
                    crop_paths
                )
            )

            track_results.append({
                "track_id": track_id,
                "sequence_length": len(crop_paths),
                "embedding_dimension": (
                    embeddings.shape[1]
                ),
                "timestamps": [
                    detection["timestamp"]
                    for detection in detections
                ],
                "crop_paths": crop_paths,
                "embeddings": (
                    embeddings.cpu()
                    .numpy()
                    .tolist()
                )
            })

            print(
                f"[DeepScan] Track {track_id} "
                f"complete. Sequence length: "
                f"{len(crop_paths)}"
            )

        return {
            "tracks_processed": len(track_results),
            "embedding_dimension": (
                2048
                if track_results
                else 0
            ),
            "tracks": track_results
        }