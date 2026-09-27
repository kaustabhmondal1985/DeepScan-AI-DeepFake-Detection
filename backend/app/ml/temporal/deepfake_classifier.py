import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[3]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn


class XceptionLSTMClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]

        logits = self.classifier(
            final_hidden
        )

        return logits


class DeepfakeClassifier:

    def __init__(
        self,
        model_path: str
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Deepfake model not found: "
                f"{self.model_path}"
            )

        print(
            "[DeepScan] Loading trained "
            "Xception + LSTM detector..."
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model = XceptionLSTMClassifier(
            input_dim=checkpoint.get(
                "input_dim",
                2048
            ),
            hidden_dim=checkpoint.get(
                "hidden_dim",
                256
            ),
            num_layers=checkpoint.get(
                "num_layers",
                2
            ),
            dropout=checkpoint.get(
                "dropout",
                0.3
            ),
            num_classes=2
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "[DeepScan] Detector loaded."
        )

        print(
            f"[DeepScan] Device: "
            f"{self.device}"
        )

    @torch.no_grad()
    def predict(
        self,
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:

        if not embeddings:
            raise ValueError(
                "No Xception embeddings "
                "were provided."
            )

        sequence = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if sequence.ndim != 2:
            raise ValueError(
                f"Expected embeddings with "
                f"shape (frames, 2048), "
                f"got {sequence.shape}"
            )

        if sequence.shape[1] != 2048:
            raise ValueError(
                f"Expected embedding dimension "
                f"2048, got {sequence.shape[1]}"
            )

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        logits = self.model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        real_probability = float(
            probabilities[0].item()
        )

        fake_probability = float(
            probabilities[1].item()
        )

        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )

        if predicted_class == 1:

            assessment = (
                "Potentially Manipulated"
            )

            confidence = fake_probability

        else:

            assessment = (
                "Likely Authentic"
            )

            confidence = real_probability

        return {
            "assessment": assessment,
            "predicted_class": predicted_class,
            "confidence": round(
                confidence,
                4
            ),
            "probabilities": {
                "real": round(
                    real_probability,
                    4
                ),
                "fake": round(
                    fake_probability,
                    4
                )
            },
            "frames_analyzed": int(
                sequence.shape[0]
            ),
            "embedding_dimension": int(
                sequence.shape[1]
            )
        }