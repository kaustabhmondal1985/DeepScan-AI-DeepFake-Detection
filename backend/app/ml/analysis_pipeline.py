from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np
import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms


# ============================================================
# PROJECT PATH
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = (
    BACKEND_DIR
    / "models"
    / "xception_finetuned_best.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

FRAMES_PER_VIDEO = 16

IMAGE_SIZE = 299

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# FINE-TUNED XCEPTION MODEL
# ============================================================

class FineTunedXception(nn.Module):
    """
    Fine-tuned Xception model.

    Architecture:

        Xception
            ↓
        2048-dimensional embedding
            ↓
        Dropout(0.4)
            ↓
        Linear(2048 → 256)
            ↓
        ReLU
            ↓
        Dropout(0.3)
            ↓
        Linear(256 → 2)

    Class mapping:

        0 = Real
        1 = Fake
    """

    def __init__(self):

        super().__init__()

        self.xception = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),

            nn.Linear(
                2048,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                2,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = self.xception(x)

        logits = self.classifier(
            features
        )

        return logits


# ============================================================
# DEEPSCAN PIPELINE
# ============================================================

class DeepScanPipeline:

    def __init__(self):

        print(
            "[DeepScan] Initializing "
            "fine-tuned Xception pipeline..."
        )

        print(
            f"[DeepScan] Device: {DEVICE}"
        )

        print(
            f"[DeepScan] Model path: "
            f"{MODEL_PATH}"
        )

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                "Fine-tuned Xception checkpoint "
                f"not found: {MODEL_PATH}"
            )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = FineTunedXception()

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):

            raise RuntimeError(
                "Invalid Xception checkpoint."
            )

        if "model_state_dict" not in checkpoint:

            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        state_dict = checkpoint[
            "model_state_dict"
        ]

        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        missing_keys, unexpected_keys = (
            self.model.load_state_dict(
                state_dict,
                strict=False,
            )
        )

        if missing_keys:

            print(
                "[DeepScan] WARNING: "
                f"Missing keys: {missing_keys}"
            )

        if unexpected_keys:

            print(
                "[DeepScan] WARNING: "
                f"Unexpected keys: "
                f"{unexpected_keys}"
            )

        self.model.to(
            DEVICE
        )

        self.model.eval()

        # ----------------------------------------------------
        # Checkpoint metadata
        # ----------------------------------------------------

        self.epoch = checkpoint.get(
            "epoch"
        )

        self.val_balanced_accuracy = (
            checkpoint.get(
                "val_balanced_accuracy"
            )
        )

        print(
            "[DeepScan] Fine-tuned Xception "
            "weights loaded successfully."
        )

        print(
            f"[DeepScan] Checkpoint epoch: "
            f"{self.epoch}"
        )

        print(
            "[DeepScan] Validation balanced "
            f"accuracy: "
            f"{self.val_balanced_accuracy}"
        )

        print(
            "[DeepScan] ML pipeline ready."
        )

    # ========================================================
    # PREDICT SINGLE FACE CROP
    # ========================================================

    @torch.no_grad()
    def predict_face(
        self,
        crop_path: str,
    ) -> Dict[str, float]:

        if not crop_path:

            raise ValueError(
                "Crop path is empty."
            )

        # ----------------------------------------------------
        # Resolve path
        # ----------------------------------------------------

        path = Path(
            crop_path
        )

        if not path.is_absolute():

            path = (
                BACKEND_DIR
                / path
            )

        if not path.exists():

            raise FileNotFoundError(
                f"Face crop not found: {path}"
            )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        tensor = TRANSFORM(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Xception inference
        # ----------------------------------------------------

        logits = self.model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        real_probability = float(
            probabilities[0][0]
            .cpu()
            .item()
        )

        fake_probability = float(
            probabilities[0][1]
            .cpu()
            .item()
        )

        return {
            "real": real_probability,
            "fake": fake_probability,
        }

    # ========================================================
    # EXTRACT USABLE FACE FRAMES
    # ========================================================

    def extract_face_frames(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        usable_frames = []

        for detection in detections:

            timestamp = float(
                detection.get(
                    "timestamp",
                    0.0,
                )
            )

            faces_container = (
                detection.get(
                    "faces"
                )
            )

            if not isinstance(
                faces_container,
                dict,
            ):
                continue

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Actual structure:
            #
            # faces:
            # {
            #     "frame_path": "...",
            #     "face_count": 1,
            #     "faces": [
            #         {
            #             "face_index": 0,
            #             "confidence": 0.999,
            #             "bbox": {...},
            #             "crop_path": "..."
            #         }
            #     ]
            # }
            # ------------------------------------------------

            face_list = (
                faces_container.get(
                    "faces",
                    [],
                )
            )

            if not isinstance(
                face_list,
                list,
            ):
                continue

            if not face_list:
                continue

            # ------------------------------------------------
            # Select largest face
            # ------------------------------------------------

            valid_faces = []

            for face in face_list:

                if not isinstance(
                    face,
                    dict,
                ):
                    continue

                bbox = face.get(
                    "bbox"
                )

                crop_path = face.get(
                    "crop_path"
                )

                if not bbox:
                    continue

                if not crop_path:
                    continue

                try:

                    width = float(
                        face.get(
                            "width",
                            bbox["x2"]
                            - bbox["x1"],
                        )
                    )

                    height = float(
                        face.get(
                            "height",
                            bbox["y2"]
                            - bbox["y1"],
                        )
                    )

                    area = (
                        width * height
                    )

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):

                    continue

                valid_faces.append(
                    (
                        area,
                        face,
                    )
                )

            if not valid_faces:
                continue

            valid_faces.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            _, largest_face = (
                valid_faces[0]
            )

            usable_frames.append(
                {
                    "timestamp": timestamp,
                    "crop_path": largest_face[
                        "crop_path"
                    ],
                    "bbox": largest_face.get(
                        "bbox"
                    ),
                    "confidence": largest_face.get(
                        "confidence"
                    ),
                }
            )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        usable_frames.sort(
            key=lambda item: item[
                "timestamp"
            ]
        )

        return usable_frames

    # ========================================================
    # UNIFORM SAMPLE
    # ========================================================

    def sample_frames(
        self,
        frames: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if len(frames) <= FRAMES_PER_VIDEO:

            return frames

        indices = np.linspace(
            0,
            len(frames) - 1,
            num=FRAMES_PER_VIDEO,
            dtype=int,
        )

        return [
            frames[int(index)]
            for index in indices
        ]

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        preprocessing_json: str,
    ) -> Dict[str, Any]:

        preprocessing_path = Path(
            preprocessing_json
        )

        if not preprocessing_path.exists():

            raise FileNotFoundError(
                "Preprocessing result not found: "
                f"{preprocessing_path}"
            )

        print(
            "[DeepScan] Loading preprocessing "
            "results..."
        )

        with open(
            preprocessing_path,
            "r",
            encoding="utf-8",
        ) as f:

            preprocessing = json.load(
                f
            )

        detections = (
            preprocessing.get(
                "detections",
                []
            )
        )

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if not detections:

            return {
                "assessment": "Inconclusive",
                "confidence": 0.0,
                "probabilities": {
                    "real": 0.0,
                    "fake": 0.0,
                },
                "reason": (
                    "No usable face detections "
                    "were available."
                ),
                "frames_analyzed": 0,
                "tracks_analyzed": 0,
            }

        # ----------------------------------------------------
        # Extract actual face crops
        # ----------------------------------------------------

        face_frames = (
            self.extract_face_frames(
                detections
            )
        )

        print(
            "[DeepScan] Usable face frames: "
            f"{len(face_frames)}"
        )

        if not face_frames:

            return {
                "assessment": "Inconclusive",
                "confidence": 0.0,
                "probabilities": {
                    "real": 0.0,
                    "fake": 0.0,
                },
                "reason": (
                    "Face detections were found, "
                    "but no usable face crops "
                    "could be created."
                ),
                "frames_analyzed": 0,
                "tracks_analyzed": 0,
                "model": {
                    "name": "Fine-tuned Xception",
                    "architecture": (
                        "Xception + MLP classifier"
                    ),
                    "input_size": "299x299",
                    "embedding_dimension": 2048,
                    "frames_per_video": (
                        FRAMES_PER_VIDEO
                    ),
                    "aggregation": (
                        "Mean frame probability"
                    ),
                },
            }

        # ----------------------------------------------------
        # Select 8 representative face crops
        # ----------------------------------------------------

        selected_frames = (
            self.sample_frames(
                face_frames
            )
        )

        print(
            "[DeepScan] Selected "
            f"{len(selected_frames)} "
            "face frames for inference."
        )

        # ----------------------------------------------------
        # Run Xception
        # ----------------------------------------------------

        frame_predictions = []

        for index, frame in enumerate(
            selected_frames
        ):

            crop_path = frame[
                "crop_path"
            ]

            print(
                "[DeepScan] Xception inference "
                f"{index + 1}/"
                f"{len(selected_frames)}: "
                f"{crop_path}"
            )

            try:

                prediction = (
                    self.predict_face(
                        crop_path
                    )
                )

                frame_predictions.append(
                    {
                        "frame_index": index,
                        "timestamp": frame[
                            "timestamp"
                        ],
                        "crop_path": crop_path,
                        "face_confidence": frame.get(
                            "confidence"
                        ),
                        "probabilities": prediction,
                    }
                )

                print(
                    "[DeepScan]   Real: "
                    f"{prediction['real']:.4f} | "
                    "Fake: "
                    f"{prediction['fake']:.4f}"
                )

            except Exception as exc:

                print(
                    "[DeepScan] Xception failed "
                    f"for frame {index}: "
                    f"{exc}"
                )

        # ----------------------------------------------------
        # No predictions
        # ----------------------------------------------------

        if not frame_predictions:

            return {
                "assessment": "Inconclusive",
                "confidence": 0.0,
                "probabilities": {
                    "real": 0.0,
                    "fake": 0.0,
                },
                "reason": (
                    "Face crops were available, "
                    "but Xception inference failed."
                ),
                "frames_analyzed": 0,
                "tracks_analyzed": 0,
                "model": {
                    "name": "Fine-tuned Xception",
                    "architecture": (
                        "Xception + MLP classifier"
                    ),
                    "input_size": "299x299",
                    "embedding_dimension": 2048,
                    "frames_per_video": (
                        FRAMES_PER_VIDEO
                    ),
                    "aggregation": (
                        "Mean frame probability"
                    ),
                },
            }

        # ----------------------------------------------------
        # Aggregate predictions
        # ----------------------------------------------------

        real_scores = [
            item[
                "probabilities"
            ]["real"]
            for item
            in frame_predictions
        ]

        fake_scores = [
            item[
                "probabilities"
            ]["fake"]
            for item
            in frame_predictions
        ]

        average_real = (
            sum(real_scores)
            / len(real_scores)
        )

        average_fake = (
            sum(fake_scores)
            / len(fake_scores)
        )

        # ----------------------------------------------------
        # Assessment
        # ----------------------------------------------------

        if average_fake >= average_real:

            assessment = (
                "Potentially Manipulated"
            )

            confidence = average_fake

        else:

            assessment = (
                "Likely Authentic"
            )

            confidence = average_real

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {

            "assessment": assessment,

            "confidence": round(
                confidence,
                4,
            ),

            "probabilities": {
                "real": round(
                    average_real,
                    4,
                ),
                "fake": round(
                    average_fake,
                    4,
                ),
            },

            "frames_analyzed": len(
                frame_predictions
            ),

            "tracks_analyzed": 1,

            "tracks": [
                {
                    "track_id": 0,
                    "frames_used": len(
                        frame_predictions
                    ),
                    "timestamps": [
                        item[
                            "timestamp"
                        ]
                        for item
                        in frame_predictions
                    ],
                    "prediction": {
                        "real": round(
                            average_real,
                            4,
                        ),
                        "fake": round(
                            average_fake,
                            4,
                        ),
                    },
                }
            ],

            "model": {

                "name": (
                    "Fine-tuned Xception"
                ),

                "architecture": (
                    "Xception + MLP classifier"
                ),

                "input_size": (
                    "299x299"
                ),

                "embedding_dimension": 2048,

                "frames_per_video": (
                    FRAMES_PER_VIDEO
                ),

                "aggregation": (
                    "Mean frame probability"
                ),

                "checkpoint": (
                    "xception_finetuned_best.pth"
                ),

                "checkpoint_epoch": (
                    self.epoch
                ),

                "checkpoint_val_balanced_accuracy": (
                    self.val_balanced_accuracy
                ),
            },

            "reason": (
                "Prediction aggregated from "
                f"{len(frame_predictions)} "
                "face crops using the "
                "fine-tuned Xception model."
            ),

            "frame_predictions": (
                frame_predictions
            ),
        }

        print(
            "[DeepScan] Xception inference "
            "completed successfully."
        )

        print(
            "[DeepScan] Average Real: "
            f"{average_real:.4f}"
        )

        print(
            "[DeepScan] Average Fake: "
            f"{average_fake:.4f}"
        )

        print(
            "[DeepScan] Assessment: "
            f"{assessment}"
        )

        return result