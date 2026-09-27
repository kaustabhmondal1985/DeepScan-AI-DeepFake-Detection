import os
from typing import List

import torch
import torch.nn as nn
from PIL import Image
import timm
from torchvision import transforms


class XceptionFeatureExtractor:
    """
    Xception-based visual feature extractor.

    This module currently extracts visual embeddings from
    detected face crops.

    IMPORTANT:
    The ImageNet-pretrained Xception model is NOT itself
    a deepfake detector. A deepfake-trained checkpoint will
    be added later for actual classification.
    """

    def __init__(
        self,
        device: str | None = None
    ):
        # --------------------------------------------------
        # Select device
        # --------------------------------------------------

        if device is None:

            if torch.cuda.is_available():
                device = "cuda"

            else:
                device = "cpu"

        self.device = torch.device(device)

        print(
            f"[DeepScan] Xception device: "
            f"{self.device}"
        )

        # --------------------------------------------------
        # Load Xception
        # --------------------------------------------------

        print(
            "[DeepScan] Loading Xception model..."
        )

        self.model = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # --------------------------------------------------
        # Image preprocessing
        # --------------------------------------------------

        self.transform = transforms.Compose([

            transforms.Resize(
                (299, 299)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        print(
            "[DeepScan] Xception model ready."
        )

    @torch.no_grad()
    def extract_embedding(
        self,
        image_path: str
    ) -> torch.Tensor:
        """
        Extract a visual embedding from one face crop.

        Returns:
            Tensor containing the Xception embedding.
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Face crop not found: {image_path}"
            )

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # --------------------------------------------------
        # Apply preprocessing
        # --------------------------------------------------

        tensor = self.transform(
            image
        )

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        # --------------------------------------------------
        # Xception inference
        # --------------------------------------------------

        embedding = self.model(
            tensor
        )

        return embedding.squeeze(0)

    @torch.no_grad()
    def extract_embeddings(
        self,
        image_paths: List[str]
    ) -> torch.Tensor:
        """
        Extract embeddings from multiple face crops.

        Returns:
            Tensor of shape:

            [number_of_faces, embedding_dimension]
        """

        if not image_paths:
            return torch.empty(
                (0, 2048)
            )

        embeddings = []

        for index, image_path in enumerate(
            image_paths
        ):

            print(
                f"[DeepScan] Xception processing "
                f"face {index + 1}/"
                f"{len(image_paths)}"
            )

            embedding = self.extract_embedding(
                image_path
            )

            embeddings.append(
                embedding
            )

        return torch.stack(
            embeddings
        )