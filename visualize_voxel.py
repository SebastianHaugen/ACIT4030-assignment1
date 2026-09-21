"""
Visualize the voxel grid representation for one example per class --
pairs nicely with visualize_classes.py (mesh renders) to show the same
objects across different input representations for your report. 

Usage:
    python visualize_voxels.py
Outputs: voxel_<class_id>.png in the current folder.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

from dataset import build_file_index, read_binvox


ROOT = "./ShapeNetCore"
VARIANT = "solid"  # matches what ShapeNetVoxel trains on


def visualize_voxel_grid(voxel_array, title, save_path):
    filled = np.argwhere(voxel_array > 0)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(filled[:, 0], filled[:, 1], filled[:, 2], zdir='z',
               c='red', marker='o', alpha=0.5, s=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


items, class_to_idx = build_file_index(ROOT)
idx_to_class = {v: k for k, v in class_to_idx.items()}

seen = set()
for mesh_path, label in items:
    class_name = idx_to_class[label]
    if class_name in seen:
        continue

    binvox_path = mesh_path.replace(".obj", f".{VARIANT}.binvox")
    try:
        vox = read_binvox(binvox_path)
    except (IOError, FileNotFoundError):
        continue  # this particular model is missing the file, try the next one

    seen.add(class_name)
    visualize_voxel_grid(vox, f"Class {class_name} (voxel)", f"voxel_{class_name}.png")

    if len(seen) == len(class_to_idx):
        break

print("Done. Open the PNG files to see one voxelized example per class.")