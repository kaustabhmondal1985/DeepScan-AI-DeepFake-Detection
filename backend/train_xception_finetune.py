import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import timm


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_ROOT = Path("training/processed")
MODEL_DIR = Path("models")

BATCH_SIZE = 8
EPOCHS = 5

# Small learning rates because pretrained Xception is being
# fine-tuned rather than trained from scratch.
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4

IMAGE_SIZE = 299

# Only the final Xception stages are trainable.
TRAINABLE_BLOCKS = 2

PATIENCE = 2


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.RandomHorizontalFlip(
        p=0.5
    ),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05,
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
])


eval_transform = transforms.Compose([
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
])


# ============================================================
# DATASET
# ============================================================

class DeepfakeFaceDataset(Dataset):

    def __init__(
        self,
        split,
        transform=None,
    ):
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = DATA_ROOT / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset directory: {split_dir}"
            )

        video_dirs = sorted(
            [
                p
                for p in split_dir.iterdir()
                if p.is_dir()
            ]
        )

        for video_dir in video_dirs:

            metadata_candidates = [
                video_dir / "metadata.json",
                video_dir / "preprocessing.json",
            ]

            metadata_path = None

            for candidate in metadata_candidates:
                if candidate.exists():
                    metadata_path = candidate
                    break

            if metadata_path is None:
                continue

            try:
                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    metadata = json.load(file)
            except Exception as exc:
                print(
                    f"[WARNING] Could not read "
                    f"{metadata_path}: {exc}"
                )
                continue

            label = metadata.get("label")

            if label is None:
                label_name = str(
                    metadata.get(
                        "label_name",
                        ""
                    )
                ).lower()

                if label_name == "real":
                    label = 0

                elif label_name == "fake":
                    label = 1

                else:
                    print(
                        f"[WARNING] No valid label: "
                        f"{metadata_path}"
                    )
                    continue

            label = int(label)

            frames = metadata.get(
                "frames",
                []
            )

            for frame in frames:

                crop_path = (
                    frame.get("crop_path")
                    or frame.get("face_crop_path")
                )

                face_detected = frame.get(
                    "face_detected",
                    True,
                )

                if not face_detected:
                    continue

                if not crop_path:
                    continue

                crop_path = Path(crop_path)

                if not crop_path.exists():

                    candidate = (
                        Path.cwd()
                        / crop_path
                    )

                    if candidate.exists():
                        crop_path = candidate

                    else:

                        # Try relative to backend.
                        candidate = (
                            DATA_ROOT.parent.parent
                            / crop_path
                        )

                        if candidate.exists():
                            crop_path = candidate

                        else:
                            continue

                self.samples.append({
                    "path": crop_path,
                    "label": label,
                    "video": metadata.get(
                        "video_name",
                        video_dir.name,
                    ),
                    "source_group": metadata.get(
                        "source_group",
                        "",
                    ),
                })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No usable face crops found "
                f"for split: {split}"
            )

        labels = [
            sample["label"]
            for sample in self.samples
        ]

        counts = Counter(labels)

        print(
            f"\n[{split.upper()}]"
        )
        print(
            f"Videos: {len(video_dirs)}"
        )
        print(
            f"Face crops: {len(self.samples)}"
        )
        print(
            f"Real crops: {counts.get(0, 0)}"
        )
        print(
            f"Fake crops: {counts.get(1, 0)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(
            sample["path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return image, label


# ============================================================
# MODEL
# ============================================================

class XceptionFineTune(nn.Module):

    def __init__(self):

        super().__init__()

        print(
            "\n[DeepScan] Loading "
            "ImageNet-pretrained Xception..."
        )

        self.xception = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # Freeze everything first.
        for parameter in self.xception.parameters():
            parameter.requires_grad = False

        # timm Xception exposes blocks through .blocks.
        if hasattr(self.xception, "blocks"):

            blocks = list(
                self.xception.blocks
            )

            number_to_train = min(
                TRAINABLE_BLOCKS,
                len(blocks),
            )

            for block in blocks[
                -number_to_train:
            ]:

                for parameter in block.parameters():
                    parameter.requires_grad = True

            print(
                f"[DeepScan] Fine-tuning "
                f"last {number_to_train} Xception blocks."
            )

        else:
            # Fallback: unfreeze the final
            # portion of the network.
            parameters = list(
                self.xception.parameters()
            )

            cutoff = int(
                len(parameters) * 0.8
            )

            for parameter in parameters[cutoff:]:
                parameter.requires_grad = True

            print(
                "[DeepScan] Using fallback "
                "partial fine-tuning."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(
                p=0.4
            ),
            nn.Linear(
                2048,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(
                p=0.3
            ),
            nn.Linear(
                256,
                2,
            ),
        )

    def forward(self, x):

        features = self.xception(x)

        return self.classifier(
            features
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    predictions,
):

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            targets,
            predictions,
        )
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    real_recall = (
        matrix[0, 0]
        / max(
            matrix[0].sum(),
            1,
        )
    )

    fake_recall = (
        matrix[1, 1]
        / max(
            matrix[1].sum(),
            1,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy":
            balanced_accuracy,
        "real_recall":
            real_recall,
        "fake_recall":
            fake_recall,
        "confusion_matrix":
            matrix,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    dataloader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    targets = []
    predictions = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            targets.extend(
                labels.cpu().numpy()
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(dataloader.dataset)
    )

    metrics = calculate_metrics(
        targets,
        predictions,
    )

    metrics["loss"] = average_loss

    return metrics


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
):

    model.train()

    total_loss = 0.0

    targets = []
    predictions = []

    for batch_index, (
        images,
        labels,
    ) in enumerate(dataloader):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predicted = torch.argmax(
            logits,
            dim=1,
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        predictions.extend(
            predicted.detach()
            .cpu()
            .numpy()
        )

        if (
            batch_index + 1
        ) % 10 == 0:

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(dataloader)}",
                end="\r",
            )

    print()

    average_loss = (
        total_loss
        / len(dataloader.dataset)
    )

    metrics = calculate_metrics(
        targets,
        predictions,
    )

    metrics["loss"] = average_loss

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics,
):

    matrix = metrics[
        "confusion_matrix"
    ]

    print(
        f"{name}: "
        f"Loss={metrics['loss']:.4f} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"BalAcc={metrics['balanced_accuracy']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"RealR={metrics['real_recall']:.4f} | "
        f"FakeR={metrics['fake_recall']:.4f}"
    )

    print(
        "  Confusion Matrix:"
    )

    print(matrix)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DeepScan - Xception Fine-Tuning"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Backbone LR: {BACKBONE_LR}"
    )

    print(
        f"Classifier LR: {CLASSIFIER_LR}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_dataset = DeepfakeFaceDataset(
        "train",
        train_transform,
    )

    val_dataset = DeepfakeFaceDataset(
        "val",
        eval_transform,
    )

    test_dataset = DeepfakeFaceDataset(
        "test",
        eval_transform,
    )

    # --------------------------------------------------------
    # BALANCED SAMPLER
    # --------------------------------------------------------

    train_labels = [
        sample["label"]
        for sample in train_dataset.samples
    ]

    counts = Counter(
        train_labels
    )

    real_count = counts.get(
        0,
        0,
    )

    fake_count = counts.get(
        1,
        0,
    )

    class_weights = {
        0: 1.0 / real_count,
        1: 1.0 / fake_count,
    }

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = XceptionFineTune()

    model = model.to(
        DEVICE
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    total_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_count:,}"
    )

    print(
        f"Total parameters: "
        f"{total_count:,}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We use a balanced sampler but NORMAL loss.
    # We do NOT combine sampler + class weights because
    # that caused collapse in the earlier experiment.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    backbone_parameters = []
    classifier_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.startswith(
            "xception."
        ):
            backbone_parameters.append(
                parameter
            )

        else:
            classifier_parameters.append(
                parameter
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    backbone_parameters,
                "lr":
                    BACKBONE_LR,
            },
            {
                "params":
                    classifier_parameters,
                "lr":
                    CLASSIFIER_LR,
            },
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_score = -1.0
    best_epoch = 0
    patience_counter = 0

    best_model_path = (
        MODEL_DIR
        / "xception_finetuned_best.pth"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 70)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        print()
        print_metrics(
            "TRAIN",
            train_metrics,
        )

        print_metrics(
            "VAL",
            val_metrics,
        )

        score = (
            val_metrics[
                "balanced_accuracy"
            ]
        )

        scheduler.step(
            score
        )

        if score > best_score:

            best_score = score
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "val_balanced_accuracy":
                        score,
                    "model":
                        "Xception fine-tuned",
                },
                best_model_path,
            )

            print(
                "\n[DeepScan] NEW BEST MODEL"
            )

            print(
                f"Validation Balanced "
                f"Accuracy: {score:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"\n[DeepScan] "
                f"No improvement "
                f"({patience_counter}/"
                f"{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print(
                    "\n[DeepScan] "
                    "Early stopping."
                )

                break

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LOADING BEST MODEL"
    )
    print("=" * 70)

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation "
        f"balanced accuracy: "
        f"{checkpoint['val_balanced_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    final_val = evaluate(
        model,
        val_loader,
        criterion,
    )

    print()
    print("=" * 70)
    print(
        "FINAL VALIDATION"
    )
    print("=" * 70)

    print_metrics(
        "VAL",
        final_val,
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    final_test = evaluate(
        model,
        test_loader,
        criterion,
    )

    print()
    print("=" * 70)
    print(
        "FINAL TEST"
    )
    print("=" * 70)

    print_metrics(
        "TEST",
        final_test,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = {
        "model":
            "Xception Fine-Tuned",
        "device":
            str(DEVICE),
        "epochs":
            EPOCHS,
        "best_epoch":
            best_epoch,
        "batch_size":
            BATCH_SIZE,
        "backbone_learning_rate":
            BACKBONE_LR,
        "classifier_learning_rate":
            CLASSIFIER_LR,
        "train_videos":
            108,
        "validation_videos":
            36,
        "test_videos":
            36,
        "best_validation_balanced_accuracy":
            best_score,
        "validation": {
            "loss":
                final_val["loss"],
            "accuracy":
                final_val["accuracy"],
            "balanced_accuracy":
                final_val[
                    "balanced_accuracy"
                ],
            "precision":
                final_val["precision"],
            "recall":
                final_val["recall"],
            "f1":
                final_val["f1"],
            "real_recall":
                final_val["real_recall"],
            "fake_recall":
                final_val["fake_recall"],
            "confusion_matrix":
                final_val[
                    "confusion_matrix"
                ].tolist(),
        },
        "test": {
            "loss":
                final_test["loss"],
            "accuracy":
                final_test["accuracy"],
            "balanced_accuracy":
                final_test[
                    "balanced_accuracy"
                ],
            "precision":
                final_test["precision"],
            "recall":
                final_test["recall"],
            "f1":
                final_test["f1"],
            "real_recall":
                final_test["real_recall"],
            "fake_recall":
                final_test["fake_recall"],
            "confusion_matrix":
                final_test[
                    "confusion_matrix"
                ].tolist(),
        },
    }

    results_path = (
        MODEL_DIR
        / "xception_finetuned_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print(
        "FINE-TUNING COMPLETED"
    )
    print("=" * 70)

    print(
        f"Model: "
        f"{best_model_path}"
    )

    print(
        f"Results: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()