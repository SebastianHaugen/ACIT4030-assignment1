"""Quick sanity check: run this first, before training.
    python smoke_test.py
"""
from dataset import build_file_index, ShapeNetPointCloud, ShapeNetVoxel, ShapeNetMesh, mesh_collate_fn
from torch.utils.data import DataLoader

ROOT = "./ShapeNetCore"

items, class_to_idx = build_file_index(ROOT)
print(f"Found {len(items)} models across classes: {class_to_idx}")
assert len(items) > 0, "No .obj files found -- check ROOT path and folder structure."

sample_items = items[:8]

print("\n--- Point cloud ---")
pc_ds = ShapeNetPointCloud(ROOT, num_points=1024, split_items=sample_items)
pts, label = pc_ds[0]
print("shape:", pts.shape, "label:", label)  # expect (1024, 3)

print("\n--- Voxel ---")
vox_ds = ShapeNetVoxel(ROOT, resolution=32, split_items=sample_items)
vox, label = vox_ds[0]
print("shape:", vox.shape, "label:", label, "occupied voxels:", vox.sum().item())  # expect (1, 32, 32, 32)

print("\n--- Mesh (PyTorch3D) ---")
mesh_ds = ShapeNetMesh(ROOT, split_items=sample_items)
loader = DataLoader(mesh_ds, batch_size=4, collate_fn=mesh_collate_fn)
meshes, labels = next(iter(loader))
print("batch of", len(meshes), "meshes, labels:", labels)
print("verts_packed shape:", meshes.verts_packed().shape)
print("edges_packed shape:", meshes.edges_packed().shape)

print("\nAll smoke tests passed.")