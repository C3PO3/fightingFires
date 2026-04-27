import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import PointNet2SemSeg
from dataset import PointCloudDataset


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (points, labels) in enumerate(loader):
        # print(f"[Train] batch {batch_idx} loaded")

        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred = model(points)

        pred = pred.permute(0, 2, 1)
        loss = criterion(pred, labels)

        loss.backward()

        optimizer.step()
        # print(f"[Train] batch {batch_idx} optimizer step done")

        total_loss += loss.item()

        pred_label = pred.argmax(dim=1)
        total_correct += (pred_label == labels).sum().item()
        total_seen += labels.numel()

    avg_loss = total_loss / len(loader)
    acc = total_correct / total_seen
    return avg_loss, acc


def compute_metrics_from_confusion(confusion):
    """
    confusion: [num_classes, num_classes]
               rows = ground truth
               cols = prediction
               
               Example: confusion[1, 2] = 7
               Actually class 1, but predicted to be class 2
    """
    confusion = confusion.astype(np.int64)

    tp = np.diag(confusion)                    # true positives for each class
    gt_count = confusion.sum(axis=1)          # total ground truth points per class
    pred_count = confusion.sum(axis=0)        # total predicted points per class

    # Per-class accuracy = TP / GT
    per_class_acc = np.full(len(tp), np.nan, dtype=np.float64)
    valid_acc = gt_count > 0
    per_class_acc[valid_acc] = tp[valid_acc] / gt_count[valid_acc]

    # Per-class IoU = TP / (GT + Pred - TP)
    union = gt_count + pred_count - tp
    per_class_iou = np.full(len(tp), np.nan, dtype=np.float64)
    valid_iou = union > 0
    per_class_iou[valid_iou] = tp[valid_iou] / union[valid_iou]

    # Mean class accuracy
    mAcc = np.nanmean(per_class_acc)

    # Mean IoU
    mIoU = np.nanmean(per_class_iou)

    # Overall accuracy
    OA = tp.sum() / confusion.sum() if confusion.sum() > 0 else 0.0

    return {
        "per_class_acc": per_class_acc,
        "per_class_iou": per_class_iou,
        "mAcc": mAcc,
        "mIoU": mIoU,
        "OA": OA,
    }

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0

    # confusion[gt, pred]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)

        pred = model(points)         # expected: [B, N, num_classes]
        pred = pred.permute(0, 2, 1) # [B, num_classes, N]

        loss = criterion(pred, labels)
        total_loss += loss.item()

        pred_label = pred.argmax(dim=1)   # [B, N]

        pred_np = pred_label.cpu().numpy().reshape(-1)
        labels_np = labels.cpu().numpy().reshape(-1)

        # update confusion matrix
        for gt, pd in zip(labels_np, pred_np):
            confusion[gt, pd] += 1

    avg_loss = total_loss / len(loader)

    metrics = compute_metrics_from_confusion(confusion)

    return avg_loss, metrics, confusion

def get_class_weights(labels_paths, num_classes, device):
    all_counts = np.zeros(num_classes, dtype=np.float32)

    for path in labels_paths:
        labels = np.load(path)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        all_counts += counts

    freq = all_counts / all_counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32, device=device)

def main():

    train_scenes = [    
            ("data/area1/points_area1.npy", "data/area1/labels_area1.npy"),
            # ("data/area4/points_area4.npy", "data/area4/labels_area4.npy"),
            ("data/area5a/points_area5a.npy", "data/area5a/labels_area5a.npy"),
            ("data/area5b/points_area5b.npy", "data/area5b/labels_area5b.npy"),
        ]
    val_scenes = [
            ("data/area3/points_area3.npy", "data/area3/labels_area3.npy"),
        ]

    train_label_paths = [
        "data/area1/labels_area1.npy",
        # "data/area4/labels_area4.npy",
        "data/area5a/labels_area5a.npy",
        "data/area5b/labels_area5b.npy"
    ]

    # hyperparameters
    batch_size = 4
    epochs = 50
    lr = 1e-3

    block_size = 1.0
    num_points = 4096
    min_points = 100
    train_num_blocks = 500
    val_num_blocks = 500


    # whether to use augmentation
    train_augment = False
    val_augment = False

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # datasets
    train_dataset = PointCloudDataset(
        train_scenes,
        block_size=block_size,
        num_points=num_points,
        min_points=min_points,
        num_blocks=train_num_blocks,
        augment=train_augment,
        fixed=False
    )

    val_dataset = PointCloudDataset(
        val_scenes,
        block_size=block_size,
        num_points=num_points,
        min_points=min_points,
        num_blocks=val_num_blocks,
        augment=val_augment,
        fixed=True
    )

    print("\n==== RAW DATASET DISTRIBUTION ====")
    train_dataset._print_class_distribution("Train (raw)")
    val_dataset._print_class_distribution("Val (raw)")

    print("\n==== SAMPLED BLOCK DISTRIBUTION ====")
    train_dataset.estimate_block_distribution(num_samples=200)
    val_dataset.estimate_block_distribution(num_samples=200)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # infer number of classes
    num_classes = 13
    class_weights = get_class_weights(train_label_paths, num_classes, device)

    # infer input feature dimension
    sample_points, _ = train_dataset[0]
    in_channels = sample_points.shape[1]
    # print("Input channels:", in_channels)

    # model / loss / optimizer
    model = PointNet2SemSeg(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_miou = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_metrics, confusion = eval_one_epoch(
            model, val_loader, criterion, device, num_classes
        )

        val_oa = val_metrics["OA"]
        val_macc = val_metrics["mAcc"]
        val_miou = val_metrics["mIoU"]
        per_class_acc = val_metrics["per_class_acc"]
        per_class_iou = val_metrics["per_class_iou"]

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train OA: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | "
            f"Val mAcc: {val_macc:.4f} | Val OA: {val_oa:.4f}"
        )

        print("Per-class metrics:")
        for cls in range(num_classes):
            acc_cls = per_class_acc[cls]
            iou_cls = per_class_iou[cls]

            acc_str = "N/A" if np.isnan(acc_cls) else f"{acc_cls:.4f}"
            iou_str = "N/A" if np.isnan(iou_cls) else f"{iou_cls:.4f}"

            print(f"  Class {cls}: Acc = {acc_str} | IoU = {iou_str}")

        print("Confusion Matrix:")
        print(confusion)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model to best_model.pth")

        print("Best val mIoU:", best_val_miou)
    print("Training finished.")


if __name__ == "__main__":
    main()