import sys
from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# =========================================================
# PROJECT PATH
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =========================================================
# PATHS
# =========================================================

FEATURES_DIR = (
    BACKEND_DIR
    / "training"
    / "features"
)

SPLITS_DIR = (
    BACKEND_DIR
    / "training"
    / "splits"
)

MODELS_DIR = (
    BACKEND_DIR
    / "models"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 7

INPUT_DIM = 2048

NUM_CLASSES = 2

EXPECTED_FRAMES = 8


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATASET
# =========================================================

class XceptionBaselineFeatureDataset(Dataset):

    def __init__(self, split):

        self.split = split

        split_file = (
            SPLITS_DIR
            / f"{split}.csv"
        )

        if not split_file.exists():

            raise FileNotFoundError(
                f"Split file not found:\n"
                f"{split_file}"
            )

        self.records = []

        with open(
            split_file,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                video_name = row["video_name"]

                label = int(row["label"])

                manipulation = row["manipulation"]

                feature_name = (
                    f"{manipulation}_"
                    f"{Path(video_name).stem}.npz"
                )

                feature_path = (
                    FEATURES_DIR
                    / split
                    / feature_name
                )

                if feature_path.exists():

                    self.records.append(
                        {
                            "feature_path":
                                feature_path,

                            "label":
                                label,

                            "video_name":
                                video_name,

                            "manipulation":
                                manipulation
                        }
                    )

                else:

                    print(
                        "[WARNING] Feature file "
                        f"missing: {feature_path}"
                    )

        if not self.records:

            raise RuntimeError(
                f"No feature files found "
                f"for split '{split}'.\n"
                f"Expected directory:\n"
                f"{FEATURES_DIR / split}"
            )

        print(
            f"[DeepScan] {split.upper()} "
            f"dataset loaded: "
            f"{len(self.records)} videos"
        )

        self.labels = np.array(
            [
                record["label"]
                for record in self.records
            ],
            dtype=np.int64
        )

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        data = np.load(
            record["feature_path"]
        )

        embeddings = np.asarray(
            data["embeddings"],
            dtype=np.float32
        )

        if embeddings.ndim != 2:

            raise ValueError(
                f"Unexpected embedding shape "
                f"{embeddings.shape} in "
                f"{record['feature_path']}"
            )

        if embeddings.shape[1] != INPUT_DIM:

            raise ValueError(
                f"Expected embedding dimension "
                f"{INPUT_DIM}, got "
                f"{embeddings.shape[1]} in "
                f"{record['feature_path']}"
            )

        if embeddings.shape[0] != EXPECTED_FRAMES:

            raise ValueError(
                f"Expected "
                f"{EXPECTED_FRAMES} frames, got "
                f"{embeddings.shape[0]} in "
                f"{record['feature_path']}"
            )

        # Mean-pool the 8 temporal Xception embeddings.
        #
        # 8 frames × 2048 features
        #              ↓
        #       mean over frames
        #              ↓
        #          2048 features

        embedding = np.mean(
            embeddings,
            axis=0
        ).astype(np.float32)

        x = torch.tensor(
            embedding,
            dtype=torch.float32
        )

        y = torch.tensor(
            record["label"],
            dtype=torch.long
        )

        return x, y


# =========================================================
# MODEL
# =========================================================

class XceptionBaselineClassifier(nn.Module):

    def __init__(
        self,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                input_dim,
                num_classes
            )
        )

    def forward(self, x):

        return self.classifier(x)


# =========================================================
# BALANCED SAMPLER
# =========================================================

def create_balanced_sampler(dataset):

    labels = dataset.labels

    class_counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    if np.any(
        class_counts == 0
    ):

        raise RuntimeError(
            "Both classes must be present "
            "for balanced sampling."
        )

    class_weights = (
        1.0 / class_counts
    )

    sample_weights = np.array(
        [
            class_weights[label]
            for label in labels
        ],
        dtype=np.float64
    )

    sample_weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(

        weights=sample_weights,

        num_samples=len(
            sample_weights
        ),

        replacement=True
    )

    print(
        "\n[DeepScan] Balanced sampler enabled."
    )

    print(
        "  Training videos are "
        "resampled to reduce class imbalance."
    )

    return sampler


# =========================================================
# TRAINING
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

    loss = (
        running_loss
        / max(total, 1)
    )

    accuracy = (
        correct
        / max(total, 1)
    )

    return loss, accuracy


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

    labels_all = []

    predictions_all = []

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

            labels_all.extend(
                y.cpu()
                .numpy()
                .tolist()
            )

            predictions_all.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )

    labels = np.array(
        labels_all
    )

    predictions = np.array(
        predictions_all
    )

    total = len(labels)

    loss = (
        running_loss
        / max(total, 1)
    )

    return (
        loss,
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

    tn = int(
        np.sum(
            (labels == 0)
            &
            (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (labels == 0)
            &
            (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (labels == 1)
            &
            (predictions == 0)
        )
    )

    tp = int(
        np.sum(
            (labels == 1)
            &
            (predictions == 1)
        )
    )

    total = (
        tn
        + fp
        + fn
        + tp
    )

    accuracy = (
        (tp + tn)
        / max(total, 1)
    )

    precision = (
        tp
        /
        max(
            tp + fp,
            1
        )
    )

    fake_recall = (
        tp
        /
        max(
            tp + fn,
            1
        )
    )

    real_recall = (
        tn
        /
        max(
            tn + fp,
            1
        )
    )

    balanced_accuracy = (
        real_recall
        + fake_recall
    ) / 2.0

    f1 = (
        2.0
        * precision
        * fake_recall
        /
        max(
            precision + fake_recall,
            1e-12
        )
    )

    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "fake_recall":
            float(fake_recall),

        "real_recall":
            float(real_recall),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "f1":
            float(f1),

        "confusion_matrix": [
            [
                tn,
                fp
            ],
            [
                fn,
                tp
            ]
        ]
    }


# =========================================================
# PRINT METRICS
# =========================================================

def print_metrics(
    name,
    loss,
    metrics
):

    print()

    print(
        name
    )

    print(
        "-" * 50
    )

    print(
        f"Loss              : "
        f"{loss:.4f}"
    )

    print(
        f"Accuracy          : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision         : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Fake Recall       : "
        f"{metrics['fake_recall']:.4f}"
    )

    print(
        f"Real Recall       : "
        f"{metrics['real_recall']:.4f}"
    )

    print(
        f"F1                : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        "                 Predicted"
    )

    print(
        "                 Real    Fake"
    )

    cm = metrics[
        "confusion_matrix"
    ]

    print(
        f"Actual Real      "
        f"{cm[0][0]:5d}   "
        f"{cm[0][1]:5d}"
    )

    print(
        f"Actual Fake      "
        f"{cm[1][0]:5d}   "
        f"{cm[1][1]:5d}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=" * 70
    )

    print(
        "DeepScan - Xception Baseline "
        "Feature Training"
    )

    print(
        "=" * 70
    )

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\n[DeepScan] Device: "
        f"{device}"
    )

    print(
        "[DeepScan] This baseline uses "
        "PRECOMPUTED Xception embeddings."
    )

    print(
        "[DeepScan] No Xception forward pass "
        "is performed during training."
    )

    # =====================================================
    # DATASETS
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING FEATURE DATASETS"
    )

    print(
        "=" * 70
    )

    train_dataset = (
        XceptionBaselineFeatureDataset(
            "train"
        )
    )

    val_dataset = (
        XceptionBaselineFeatureDataset(
            "val"
        )
    )

    test_dataset = (
        XceptionBaselineFeatureDataset(
            "test"
        )
    )

    # =====================================================
    # SAMPLER
    # =====================================================

    train_sampler = (
        create_balanced_sampler(
            train_dataset
        )
    )

    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
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

    # =====================================================
    # MODEL
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INITIALIZING CLASSIFIER"
    )

    print(
        "=" * 70
    )

    print(
        "[DeepScan] Input dimension: "
        f"{INPUT_DIM}"
    )

    print(
        "[DeepScan] Temporal frames per "
        f"video: {EXPECTED_FRAMES}"
    )

    print(
        "[DeepScan] Mean pooling: "
        "8 × 2048 → 2048"
    )

    model = (
        XceptionBaselineClassifier(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES
        )
        .to(device)
    )

    # =====================================================
    # LOSS
    # =====================================================

    # Balanced sampler already compensates for the
    # 1:5 class imbalance.
    #
    # Therefore we intentionally use a normal
    # CrossEntropyLoss instead of combining sampling
    # with class-weighted loss.

    criterion = (
        nn.CrossEntropyLoss()
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    )

    # =====================================================
    # SCHEDULER
    # =====================================================

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    )

    # =====================================================
    # TRAINING
    # =====================================================

    best_val_f1 = -1.0

    best_val_balanced_accuracy = -1.0

    best_epoch = 0

    patience_counter = 0

    checkpoint_path = (
        MODELS_DIR
        / "xception_baseline_best.pth"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        (
            val_loss,
            val_labels,
            val_predictions
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        val_metrics = (
            calculate_metrics(
                val_labels,
                val_predictions
            )
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"\nEpoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Acc : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss  : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val F1    : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"Val BalAcc: "
            f"{val_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Val Real R: "
            f"{val_metrics['real_recall']:.4f}"
        )

        print(
            f"Val Fake R: "
            f"{val_metrics['fake_recall']:.4f}"
        )

        print(
            f"LR        : "
            f"{current_lr:.6f}"
        )

        # -------------------------------------------------
        # CHECKPOINT
        #
        # Primary:
        # validation balanced accuracy
        #
        # Tie-breaker:
        # validation F1
        # -------------------------------------------------

        improved = False

        if (
            val_metrics[
                "balanced_accuracy"
            ]
            > best_val_balanced_accuracy
        ):

            improved = True

        elif (
            val_metrics[
                "balanced_accuracy"
            ]
            == best_val_balanced_accuracy
            and
            val_metrics["f1"]
            > best_val_f1
        ):

            improved = True

        if improved:

            best_val_balanced_accuracy = (
                val_metrics[
                    "balanced_accuracy"
                ]
            )

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_dim":
                        INPUT_DIM,

                    "best_val_f1":
                        best_val_f1,

                    "best_val_balanced_accuracy":
                        best_val_balanced_accuracy,

                    "best_epoch":
                        best_epoch
                },
                checkpoint_path
            )

            print(
                "  -> New best model saved."
            )

        else:

            patience_counter += 1

            if (
                patience_counter
                >= PATIENCE
            ):

                print(
                    "\n[DeepScan] "
                    "Early stopping."
                )

                break

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING BEST MODEL"
    )

    print(
        "=" * 70
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"[DeepScan] Best epoch: "
        f"{checkpoint['best_epoch']}"
    )

    print(
        f"[DeepScan] Best validation F1: "
        f"{checkpoint['best_val_f1']:.4f}"
    )

    print(
        f"[DeepScan] Best validation "
        f"balanced accuracy: "
        f"{checkpoint['best_val_balanced_accuracy']:.4f}"
    )

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    (
        val_loss,
        val_labels,
        val_predictions
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    val_metrics = (
        calculate_metrics(
            val_labels,
            val_predictions
        )
    )

    print_metrics(
        "FINAL VALIDATION RESULTS",
        val_loss,
        val_metrics
    )

    # =====================================================
    # FINAL TEST
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    (
        test_loss,
        test_labels,
        test_predictions
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    test_metrics = (
        calculate_metrics(
            test_labels,
            test_predictions
        )
    )

    print_metrics(
        "FINAL TEST RESULTS",
        test_loss,
        test_metrics
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results = {

        "model":
            "Xception Baseline",

        "feature_type":
            "Precomputed ImageNet-pretrained "
            "Xception embeddings",

        "training_strategy": {

            "balanced_sampler":
                True,

            "class_weighted_loss":
                False,

            "pooling":
                "mean",

            "checkpoint_metric":
                "validation_balanced_accuracy",

            "checkpoint_tiebreaker":
                "validation_f1"
        },

        "frames_per_video":
            EXPECTED_FRAMES,

        "input_dimension":
            INPUT_DIM,

        "train_videos":
            len(train_dataset),

        "validation_videos":
            len(val_dataset),

        "test_videos":
            len(test_dataset),

        "best_epoch":
            int(
                checkpoint[
                    "best_epoch"
                ]
            ),

        "validation": {

            "loss":
                float(val_loss),

            **val_metrics
        },

        "test": {

            "loss":
                float(test_loss),

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

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "XCEPTION BASELINE TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest model:"
    )

    print(
        checkpoint_path
    )

    print(
        "\nResults:"
    )

    print(
        results_path
    )

    print(
        "\n[DeepScan] Final test "
        "balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        "[DeepScan] Final test "
        "real recall: "
        f"{test_metrics['real_recall']:.4f}"
    )

    print(
        "[DeepScan] Final test "
        "fake recall: "
        f"{test_metrics['fake_recall']:.4f}"
    )

    print(
        "\n"
        + "=" * 70
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
