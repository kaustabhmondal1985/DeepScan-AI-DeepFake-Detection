import sys
from pathlib import Path

# ---------------------------------------------------------
# Make backend/ available so "from app..." imports work
# ---------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# CONFIGURATION
# =========================================================

FEATURES_DIR = BACKEND_DIR / "training" / "features"
SPLITS_DIR = BACKEND_DIR / "training" / "splits"
MODELS_DIR = BACKEND_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

BATCH_SIZE = 16
EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 7

INPUT_DIM = 2048
HIDDEN_DIM = 512
DROPOUT = 0.3

# Number of Xception embeddings per video
EXPECTED_FRAMES = 8


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATASET
# =========================================================

class XceptionVideoDataset(Dataset):

    def __init__(self, split: str):

        self.split = split

        split_file = SPLITS_DIR / f"{split}.csv"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found:\n{split_file}"
            )

        # Read CSV manually to avoid adding another dependency
        import csv

        self.records = []

        with open(split_file, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:

                video_name = row["video_name"]
                label = int(row["label"])
                manipulation = row["manipulation"]

                # Expected feature file
                feature_name = (
                    f"{manipulation}_{Path(video_name).stem}.npz"
                )

                feature_path = (
                    FEATURES_DIR
                    / split
                    / feature_name
                )

                if feature_path.exists():

                    self.records.append(
                        {
                            "feature_path": feature_path,
                            "label": label,
                            "video_name": video_name,
                            "manipulation": manipulation,
                        }
                    )

                else:

                    print(
                        f"[WARNING] Feature file missing: "
                        f"{feature_path}"
                    )

        if not self.records:

            raise RuntimeError(
                f"No feature files found for split '{split}'.\n"
                f"Expected directory:\n"
                f"{FEATURES_DIR / split}"
            )

        print(
            f"[DeepScan] {split.upper()} dataset loaded: "
            f"{len(self.records)} videos"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        data = np.load(record["feature_path"])

        embeddings = data["embeddings"]

        # Expected:
        # (8, 2048)
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # Safety check
        if embeddings.ndim != 2:

            raise ValueError(
                f"Unexpected embedding shape "
                f"{embeddings.shape} in "
                f"{record['feature_path']}"
            )

        # -------------------------------------------------
        # Mean pooling over the 8 sampled frames
        # -------------------------------------------------
        #
        # (8, 2048)
        #      ↓
        # (2048,)
        #
        # This creates the Xception baseline representation.
        # -------------------------------------------------

        video_embedding = np.mean(
            embeddings,
            axis=0
        )

        x = torch.tensor(
            video_embedding,
            dtype=torch.float32
        )

        y = torch.tensor(
            record["label"],
            dtype=torch.long
        )

        return x, y


# =========================================================
# CLASSIFIER
# =========================================================

class XceptionClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.BatchNorm1d(
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# CLASS WEIGHTS
# =========================================================

def calculate_class_weights(dataset):

    labels = [
        record["label"]
        for record in dataset.records
    ]

    counts = np.bincount(
        labels,
        minlength=2
    )

    real_count = counts[0]
    fake_count = counts[1]

    print("\n[DeepScan] Class distribution:")

    print(
        f"  Real (0): {real_count}"
    )

    print(
        f"  Fake (1): {fake_count}"
    )

    if real_count == 0 or fake_count == 0:

        raise RuntimeError(
            "Both classes must be present in training data."
        )

    total = real_count + fake_count

    # Balanced inverse-frequency weights
    weight_real = total / (2.0 * real_count)
    weight_fake = total / (2.0 * fake_count)

    weights = torch.tensor(
        [
            weight_real,
            weight_fake
        ],
        dtype=torch.float32
    )

    print(
        "\n[DeepScan] Class weights:"
    )

    print(
        f"  Real: {weight_real:.4f}"
    )

    print(
        f"  Fake: {weight_fake:.4f}"
    )

    return weights


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * x.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += x.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            running_loss += (
                loss.item()
                * x.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_labels.extend(
                y.cpu().numpy().tolist()
            )

    total = len(all_labels)

    loss = (
        running_loss / total
    )

    predictions = np.array(
        all_predictions
    )

    labels = np.array(
        all_labels
    )

    accuracy = np.mean(
        predictions == labels
    )

    return (
        loss,
        accuracy,
        labels,
        predictions
    )


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    labels,
    predictions
):

    # Confusion matrix:
    #
    #              Predicted
    #             Real   Fake
    #
    # Actual Real  TN     FP
    #        Fake  FN     TP

    tn = np.sum(
        (labels == 0)
        & (predictions == 0)
    )

    fp = np.sum(
        (labels == 0)
        & (predictions == 1)
    )

    fn = np.sum(
        (labels == 1)
        & (predictions == 0)
    )

    tp = np.sum(
        (labels == 1)
        & (predictions == 1)
    )

    accuracy = (
        (tp + tn)
        / max(
            tp + tn + fp + fn,
            1
        )
    )

    precision = (
        tp / max(
            tp + fp,
            1
        )
    )

    recall = (
        tp / max(
            tp + fn,
            1
        )
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            precision + recall,
            1e-12
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ]
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("DeepScan - Xception Baseline Training")
    print("=" * 70)

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\n[DeepScan] Device: {device}"
    )

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    train_dataset = XceptionVideoDataset(
        "train"
    )

    val_dataset = XceptionVideoDataset(
        "val"
    )

    test_dataset = XceptionVideoDataset(
        "test"
    )

    print()

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = XceptionClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )

    model = model.to(device)

    print(
        "\n[DeepScan] Xception classifier created."
    )

    print(
        f"[DeepScan] Input dimension: {INPUT_DIM}"
    )

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset
    )

    class_weights = class_weights.to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # -----------------------------------------------------
    # Learning-rate scheduler
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    patience_counter = 0

    checkpoint_path = (
        MODELS_DIR
        / "xception_baseline_best.pth"
    )

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # -------------------------------------------------
        # Save best model according to validation loss
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": INPUT_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                checkpoint_path
            )

            print(
                f"  -> Best model saved: "
                f"{checkpoint_path}"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"\n[DeepScan] Early stopping "
                    f"at epoch {epoch}."
                )

                break

    # -----------------------------------------------------
    # Load best model
    # -----------------------------------------------------

    print(
        "\n[DeepScan] Loading best checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"[DeepScan] Best epoch: "
        f"{checkpoint['best_epoch']}"
    )

    print(
        f"[DeepScan] Best validation loss: "
        f"{checkpoint['best_val_loss']:.4f}"
    )

    # -----------------------------------------------------
    # Validation metrics
    # -----------------------------------------------------

    val_loss, val_accuracy, val_labels, val_predictions = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    val_metrics = calculate_metrics(
        val_labels,
        val_predictions
    )

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_loss, test_accuracy, test_labels, test_predictions = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_predictions
    )

    # -----------------------------------------------------
    # Print validation results
    # -----------------------------------------------------

    print("\nValidation Results")
    print("-" * 40)

    print(
        f"Loss      : {val_loss:.4f}"
    )

    print(
        f"Accuracy  : {val_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {val_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {val_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : {val_metrics['f1']:.4f}"
    )

    # -----------------------------------------------------
    # Print test results
    # -----------------------------------------------------

    print("\nTest Results")
    print("-" * 40)

    print(
        f"Loss      : {test_loss:.4f}"
    )

    print(
        f"Accuracy  : {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {test_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : {test_metrics['f1']:.4f}"
    )

    print("\nConfusion Matrix")
    print(
        "                 Predicted"
    )
    print(
        "              Real    Fake"
    )

    cm = test_metrics[
        "confusion_matrix"
    ]

    print(
        f"Actual Real   {cm[0][0]:5d}   {cm[0][1]:5d}"
    )

    print(
        f"Actual Fake   {cm[1][0]:5d}   {cm[1][1]:5d}"
    )

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    results = {
        "model": "Xception Baseline",
        "feature_type": "ImageNet-pretrained Xception embeddings",
        "frames_per_video": EXPECTED_FRAMES,
        "input_dimension": INPUT_DIM,
        "pooling": "mean",
        "train_videos": len(train_dataset),
        "validation_videos": len(val_dataset),
        "test_videos": len(test_dataset),
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "validation": {
            "loss": float(val_loss),
            **val_metrics
        },
        "test": {
            "loss": float(test_loss),
            **test_metrics
        }
    }

    results_path = (
        MODELS_DIR
        / "xception_baseline_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\n[DeepScan] Results saved to:"
    )

    print(
        results_path
    )

    print(
        f"\n[DeepScan] Model saved to:"
    )

    print(
        checkpoint_path
    )

    print("\n" + "=" * 70)
    print("XCEPTION BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()