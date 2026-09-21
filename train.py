"""
Usage:
    python train.py --model pointnet --root /path/to/ShapeNetCore --epochs 30
    python train.py --model voxel    --root /path/to/ShapeNetCore --epochs 30
    python train.py --model meshgcn  --root /path/to/ShapeNetCore --epochs 30

Splits the dataset 80/20 train/test per class (adapt as needed), trains,
and prints/saves a final accuracy + confusion matrix so you can drop the
numbers straight into the report's comparison section.
"""

import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

from dataset import (
    build_file_index, ShapeNetPointCloud, ShapeNetVoxel, ShapeNetMesh, mesh_collate_fn
)
from model_pointnet import PointNetClassifier
from model_voxel_cnn import Voxel3DCNN
from model_mesh_gcn import MeshGCN


def split_items(items, test_frac=0.2, seed=42):
    random.Random(seed).shuffle(items)
    n_test = int(len(items) * test_frac)
    return items[n_test:], items[:n_test]


def run_epoch(model, loader, device, optimizer=None, is_mesh=False):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        if is_mesh:
            meshes, labels = batch
            meshes = meshes.to(device)
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(meshes) if is_mesh else model(inputs)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(all_labels), acc, all_labels, all_preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["pointnet", "voxel", "meshgcn"], required=True)
    p.add_argument("--root", required=True, help="path to unzipped ShapeNetCore subset")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--voxel_res", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    items, class_to_idx = build_file_index(args.root)
    num_classes = len(class_to_idx)
    train_items, test_items = split_items(items)
    print(f"{len(train_items)} train / {len(test_items)} test, classes: {class_to_idx}")

    is_mesh = args.model == "meshgcn"
    collate = mesh_collate_fn if is_mesh else None

    if args.model == "pointnet":
        train_ds = ShapeNetPointCloud(args.root, num_points=args.num_points, split_items=train_items)
        test_ds = ShapeNetPointCloud(args.root, num_points=args.num_points, split_items=test_items)
        model = PointNetClassifier(num_classes=num_classes)
    elif args.model == "voxel":
        train_ds = ShapeNetVoxel(args.root, resolution=args.voxel_res, split_items=train_items)
        test_ds = ShapeNetVoxel(args.root, resolution=args.voxel_res, split_items=test_items)
        model = Voxel3DCNN(num_classes=num_classes, resolution=args.voxel_res)
    else:
        train_ds = ShapeNetMesh(args.root, split_items=train_items)
        test_ds = ShapeNetMesh(args.root, split_items=test_items)
        model = MeshGCN(num_classes=num_classes)

    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, optimizer, is_mesh)
        val_loss, val_acc, labels, preds = run_epoch(model, test_loader, device, None, is_mesh)
        best_acc = max(best_acc, val_acc)
        print(f"[{args.model}] epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")

    print(f"\nBest val accuracy for {args.model}: {best_acc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(labels, preds))
    torch.save(model.state_dict(), f"{args.model}_final.pt")


if __name__ == "__main__":
    main()