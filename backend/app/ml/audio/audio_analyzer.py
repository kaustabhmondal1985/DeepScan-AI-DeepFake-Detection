from typing import Dict, Any

import numpy as np
import torch

from app.ml.audio.audio_model import AudioCNN


class AudioAnalyzer:

    def __init__(self, embedding_size: int = 256):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(
            f"[DeepScan] Audio CNN device: {self.device}"
        )

        self.model = AudioCNN(
            embedding_size=embedding_size
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.embedding_size = embedding_size

        print(
            "[DeepScan] Audio CNN ready."
        )

    @torch.no_grad()
    def generate_embedding(
        self,
        mel_spectrogram: np.ndarray
    ) -> Dict[str, Any]:

        if mel_spectrogram.ndim != 2:
            raise ValueError(
                "Expected Mel spectrogram with shape "
                "[n_mels, time]."
            )

        # Convert NumPy array → PyTorch tensor
        tensor = torch.tensor(
            mel_spectrogram,
            dtype=torch.float32
        )

        # Add channel dimension
        # [128, 313] → [1, 128, 313]
        tensor = tensor.unsqueeze(0)

        # Add batch dimension
        # [1, 128, 313] → [1, 1, 128, 313]
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        # Run CNN
        embedding = self.model(tensor)

        # Move result back to CPU
        embedding = embedding.squeeze(0).cpu()

        return {
            "input_shape": list(mel_spectrogram.shape),
            "embedding_dimension": int(embedding.shape[0]),
            "embedding": embedding.numpy().tolist()
        }