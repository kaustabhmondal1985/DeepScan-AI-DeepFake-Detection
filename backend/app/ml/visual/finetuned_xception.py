from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from PIL import Image
import timm
from torchvision import transforms


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
    .parent
)

MODEL_PATH = (
    BACKEND_DIR
    / "models"
    / "xception_finetuned_best.pth"
)


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = 299
INPUT_DIMENSION = 2048
NUM_CLASSES = 2


# ============================================================
# MODEL
# ============================================================

class FineTunedXception(nn.Module):

    def __init__(self):

        super().__init__()

        print(
            "[DeepScan] Loading fine-tuned Xception..."
        )

        self.xception = timm.create_model(
            "xception",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        self.classifier = nn.Sequential(
            nn.Dropout(
                p=0.4
            ),

            nn.Linear(
                INPUT_DIMENSION,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                256,
                NUM_CLASSES
            ),
        )


    def forward(self, x):

        features = self.xception(x)

        logits = self.classifier(
            features
        )

        return logits


# ============================================================
# XCEPTION INFERENCE WRAPPER
# ============================================================

class FineTunedXceptionAnalyzer:

    def __init__(
        self,
        model_path: str = None
    ):

        if model_path is None:
            model_path = str(
                MODEL_PATH
            )

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Fine-tuned Xception model "
                f"not found:\n{self.model_path}"
            )

        print(
            "[DeepScan] Model path:"
        )

        print(
            f"  {self.model_path}"
        )

        print(
            f"[DeepScan] Device: {DEVICE}"
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = FineTunedXception()

        checkpoint = torch.load(
            self.model_path,
            map_location=DEVICE
        )

        # ----------------------------------------------------
        # Checkpoint format
        # ----------------------------------------------------

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model = self.model.to(
            DEVICE
        )

        self.model.eval()

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        self.transform = transforms.Compose(
            [

                transforms.Resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
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
                ),
            ]
        )

        print(
            "[DeepScan] Fine-tuned Xception "
            "model ready."
        )


    # ========================================================
    # SINGLE IMAGE
    # ========================================================

    @torch.no_grad()
    def predict_image(
        self,
        image_path: str
    ) -> Dict[str, Any]:

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Face crop not found: "
                f"{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        ).to(DEVICE)

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

        if (
            fake_probability
            >= real_probability
        ):

            prediction = (
                "Potentially Manipulated"
            )

            confidence = (
                fake_probability
            )

        else:

            prediction = (
                "Likely Authentic"
            )

            confidence = (
                real_probability
            )

        return {

            "assessment":
                prediction,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "probabilities": {

                "real":
                    round(
                        real_probability,
                        4
                    ),

                "fake":
                    round(
                        fake_probability,
                        4
                    )
            }
        }


    # ========================================================
    # MULTIPLE FACE CROPS
    # ========================================================

    @torch.no_grad()
    def predict_images(
        self,
        image_paths: List[str]
    ) -> Dict[str, Any]:

        if not image_paths:

            return {

                "assessment":
                    "Inconclusive",

                "confidence":
                    0.0,

                "probabilities": {

                    "real":
                        0.0,

                    "fake":
                        0.0
                },

                "frames_analyzed":
                    0
            }

        tensors = []

        valid_paths = []

        for image_path in image_paths:

            image_path = Path(
                image_path
            )

            if not image_path.exists():

                print(
                    "[DeepScan] WARNING: "
                    f"Crop not found: "
                    f"{image_path}"
                )

                continue

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                tensor = self.transform(
                    image
                )

                tensors.append(
                    tensor
                )

                valid_paths.append(
                    image_path
                )

            except Exception as exc:

                print(
                    "[DeepScan] WARNING: "
                    f"Could not load crop "
                    f"{image_path}: {exc}"
                )

        if not tensors:

            return {

                "assessment":
                    "Inconclusive",

                "confidence":
                    0.0,

                "probabilities": {

                    "real":
                        0.0,

                    "fake":
                        0.0
                },

                "frames_analyzed":
                    0
            }

        batch = torch.stack(
            tensors
        ).to(DEVICE)

        logits = self.model(
            batch
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # Average frame-level probabilities
        # ----------------------------------------------------

        average_probabilities = (
            probabilities.mean(
                dim=0
            )
        )

        real_probability = float(
            average_probabilities[0].item()
        )

        fake_probability = float(
            average_probabilities[1].item()
        )

        if (
            fake_probability
            >= real_probability
        ):

            assessment = (
                "Potentially Manipulated"
            )

            confidence = (
                fake_probability
            )

        else:

            assessment = (
                "Likely Authentic"
            )

            confidence = (
                real_probability
            )

        # ----------------------------------------------------
        # Individual frame predictions
        # ----------------------------------------------------

        frame_predictions = []

        for index, path in enumerate(
            valid_paths
        ):

            real_score = float(
                probabilities[
                    index,
                    0
                ].item()
            )

            fake_score = float(
                probabilities[
                    index,
                    1
                ].item()
            )

            if fake_score >= real_score:

                frame_assessment = (
                    "Potentially Manipulated"
                )

            else:

                frame_assessment = (
                    "Likely Authentic"
                )

            frame_predictions.append(
                {

                    "path":
                        str(path),

                    "assessment":
                        frame_assessment,

                    "probabilities": {

                        "real":
                            round(
                                real_score,
                                4
                            ),

                        "fake":
                            round(
                                fake_score,
                                4
                            )
                    }
                }
            )

        return {

            "assessment":
                assessment,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "probabilities": {

                "real":
                    round(
                        real_probability,
                        4
                    ),

                "fake":
                    round(
                        fake_probability,
                        4
                    )
            },

            "frames_analyzed":
                len(valid_paths),

            "frame_predictions":
                frame_predictions
        }