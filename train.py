
from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from model import UNet


# =========================
# CONFIGURATION
# =========================

DATA_DIR = Path("data/kaggle_3m")
OUTPUT_PATH = Path("artifacts/unet_brain_tumor.pth")

IMAGE_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42
VAL_RATIO = 0.20

VALID_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg"
}


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# FIND IMAGE-MASK PAIRS
# =========================

def find_pairs(data_dir):
    """
    Find image and mask pairs named:
    image.tif
    image_mask.tif

    Supports matching filenames with the same extension.
    """

    images = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        if path.stem.lower().endswith("_mask"):
            continue

        mask_path = path.with_name(
            f"{path.stem}_mask{path.suffix}"
        )

        if mask_path.exists():
            images.append((path, mask_path))

    return sorted(images)


# =========================
# DATASET
# =========================

class MRISegmentationDataset(Dataset):

    def __init__(self, pairs, image_size=IMAGE_SIZE):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

        # Load MRI image.
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        # Load ground-truth mask.
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        image = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        mask = np.asarray(
            mask, dtype=np.float32
        )

        # Binary mask: background=0, annotated region=1.
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# =========================
# LOSS FUNCTION
# =========================

def segmentation_loss(logits, targets):

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets
    )

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + 1e-7
    ) / (denominator + 1e-7)

    dice_loss = 1.0 - dice.mean()

    return bce + dice_loss


# =========================
# METRICS
# =========================

def calculate_metrics(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    dimensions = (1, 2, 3)

    intersection = (
        predictions * targets
    ).sum(dim=dimensions)

    predicted_pixels = predictions.sum(dim=dimensions)
    target_pixels = targets.sum(dim=dimensions)

    union = (
        predicted_pixels + target_pixels - intersection
    )

    dice_denominator = predicted_pixels + target_pixels

    # Empty prediction + empty target = perfect overlap.
    dice = torch.where(
        dice_denominator == 0,
        torch.ones_like(dice_denominator),
        (2.0 * intersection + 1e-7)
        / (dice_denominator + 1e-7)
    )

    iou = torch.where(
        union == 0,
        torch.ones_like(union),
        (intersection + 1e-7) / (union + 1e-7)
    )

    # Aggregate pixel counts across the batch.
    true_positive = intersection.sum()
    false_positive = (
        predicted_pixels - intersection
    ).sum()
    false_negative = (
        target_pixels - intersection
    ).sum()

    precision = (
        true_positive / (
            true_positive + false_positive + 1e-7
        )
    )

    recall = (
        true_positive / (
            true_positive + false_negative + 1e-7
        )
    )

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }


# =========================
# PATIENT-LEVEL SPLIT
# =========================

def split_by_patient(pairs):

    patient_groups = {}

    for index, (image_path, _) in enumerate(pairs):

        # Assumes the immediate parent folder
        # represents a patient.
        patient_id = image_path.parent.name

        patient_groups.setdefault(
            patient_id, []
        ).append(index)

    patient_ids = sorted(patient_groups.keys())

    if len(patient_ids) < 2:
        raise RuntimeError(
            "At least two distinct patient folders are required "
            "for a patient-level train/validation split. "
            "Check your dataset folder structure."
        )

    rng = random.Random(SEED)
    rng.shuffle(patient_ids)

    val_count = max(
        1, round(len(patient_ids) * VAL_RATIO)
    )
    val_count = min(val_count, len(patient_ids) - 1)

    validation_patients = set(patient_ids[:val_count])

    train_indices = []
    val_indices = []

    for patient_id, indices in patient_groups.items():

        if patient_id in validation_patients:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return train_indices, val_indices


# =========================
# TRAINING AND VALIDATION
# =========================

def main():

    set_seed()

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Place the extracted MRI dataset in the expected folder."
        )

    pairs = find_pairs(DATA_DIR)

    if len(pairs) < 10:
        raise RuntimeError(
            f"Found only {len(pairs)} image/mask pairs. "
            "Check the dataset filenames and folder structure."
        )

    print(f"Found {len(pairs)} image/mask pairs.")

    dataset = MRISegmentationDataset(pairs)

    train_indices, val_indices = split_by_patient(pairs)

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    print(f"Training slices: {len(train_ds)}")
    print(f"Validation slices: {len(val_ds)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = UNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    best_val_loss = float("inf")

    # Save the best checkpoint based on validation loss.
    for epoch in range(1, EPOCHS + 1):

        # -------------------------
        # TRAINING
        # -------------------------

        model.train()
        train_loss_total = 0.0

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = segmentation_loss(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        train_loss = (
            train_loss_total / len(train_loader)
        )

        # -------------------------
        # VALIDATION
        # -------------------------

        model.eval()

        val_loss_total = 0.0
        all_metrics = {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
        }

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)

                loss = segmentation_loss(logits, masks)
                val_loss_total += loss.item()

                # Calculate metrics for each image.
                metrics = calculate_metrics(logits, masks)

                # Note: batch metrics are averaged here.
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])

        val_loss = val_loss_total / len(val_loader)

        mean_metrics = {
            key: float(np.mean(values))
            for key, values in all_metrics.items()
        }

        print(
            f"\nEpoch {epoch}/{EPOCHS}\n"
            f"Train loss : {train_loss:.4f}\n"
            f"Val loss   : {val_loss:.4f}\n"
            f"Dice       : {mean_metrics['dice']:.4f}\n"
            f"IoU        : {mean_metrics['iou']:.4f}\n"
            f"Precision  : {mean_metrics['precision']:.4f}\n"
            f"Recall     : {mean_metrics['recall']:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                OUTPUT_PATH
            )

            print(f"Saved best checkpoint: {OUTPUT_PATH}")

    print("\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {OUTPUT_PATH}")

    print(
        "\nIMPORTANT: These metrics are preliminary validation "
        "results, not proof of clinical performance. Verify patient "
        "identifiers, dataset mask semantics, and use a separate "
        "unseen test set before making performance claims."
    )


if __name__ == "__main__":
    main()