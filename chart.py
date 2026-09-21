"""
Generates the validation-accuracy-vs-epoch comparison chart for the report.
Values below are taken directly from the printed training logs for the
10-epoch runs of each model.

Usage:
    python make_chart.py
Output: accuracy_curve.png (saved in the current folder)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, so this works without a display
import matplotlib.pyplot as plt

epochs = list(range(1, 11))

voxel_val    = [0.873, 0.902, 0.897, 0.907, 0.901, 0.913, 0.923, 0.915, 0.892, 0.923]
pointnet_val = [0.884, 0.909, 0.896, 0.895, 0.915, 0.910, 0.928, 0.911, 0.909, 0.884]
meshgcn_val  = [0.612, 0.735, 0.788, 0.771, 0.797, 0.801, 0.769, 0.832, 0.797, 0.808]

plt.figure(figsize=(7, 4.5))
plt.plot(epochs, voxel_val, marker='o', label='3D CNN (Voxel)', color='#2E7D32')
plt.plot(epochs, pointnet_val, marker='s', label='PointNet', color='#1565C0')
plt.plot(epochs, meshgcn_val, marker='^', label='Mesh GCN', color='#C62828')

plt.xlabel('Epoch')
plt.ylabel('Validation accuracy')
plt.title('Validation accuracy over training (10 epochs)')
plt.xticks(epochs)
plt.ylim(0.5, 1.0)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_curve.png', dpi=150)
print("Saved accuracy_curve.png")