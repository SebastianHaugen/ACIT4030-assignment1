"""
Dataset classes for the ShapeNetCore subset (5 classes).

Expected folder layout 
    root/
        <class_id_or_name>/
            <model_id>/
                model.obj / model.off        <- mesh
                model.binvox                 <- voxel grid
                (point cloud is derived from mesh, see below)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import trimesh
except ImportError:
    trimesh = None  # only needed for mesh / point cloud / voxel-from-mesh paths

try:
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes
except ImportError:
    load_obj = None  # only needed for mesh + pointnet paths


# Confirmed class (synset) IDs for this dataset:
def read_binvox(file_path):
    """
    Reads a .binvox file and returns a 3D numpy array representing the voxel grid.
    """
    with open(file_path, 'rb') as f:
        line = f.readline().decode().strip()
        if line != '#binvox 1':
            raise IOError('Not a binvox file')

        dims = []
        while True:
            line = f.readline().decode().strip()
            if line.startswith('dim'):
                dims = list(map(int, line.split()[1:]))
            elif line == 'data':
                break

        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype(np.float32)
        data = data.reshape(dims)

        return data


def build_file_index(root, classes=None):
    """Walk the dataset root and build a list of (mesh_path, label) for every model.

    Matches the standard ShapeNetCore-v2 layout confirmed for this dataset:
        root/<class_id>/<model_id>/models/model_normalized.obj
        root/<class_id>/<model_id>/models/model_normalized.solid.binvox
        root/<class_id>/<model_id>/models/model_normalized.surface.binvox
    """
    classes = classes or sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items = []
    for c in classes:
        class_dir = os.path.join(root, c)
        model_dirs = [
            os.path.join(class_dir, d)
            for d in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, d))
        ]
        for md in model_dirs:
            mesh_path = os.path.join(md, "models", "model_normalized.obj")
            if os.path.exists(mesh_path):
                items.append((mesh_path, class_to_idx[c]))
    return items, class_to_idx


# ---------------------------------------------------------------------------
# 1. Point cloud dataset (for PointNet) -- sample points off the mesh
# ---------------------------------------------------------------------------
class ShapeNetPointCloud(Dataset):
    """
    For the assignments specified conversion strategy, we're using mesh vertices
    directly as the point cloud, converted to a fixed
    size N:
        - if the mesh has more than N vertices, randomly sample N of them
        - if fewer, keep all vertices and zero-pad up to N
    This is intentionally simple (no uniform surface sampling) since the
    models are already consistently oriented and no need for rotational or
    translational invariance handling here.
    """
    def __init__(self, root, classes=None, num_points=1024, split_items=None):
        self.items, self.class_to_idx = (
            (split_items, None) if split_items is not None
            else build_file_index(root, classes)
        )
        self.num_points = num_points

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        mesh = trimesh.load(path, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float32)  # (V, 3)

        n = verts.shape[0]
        N = self.num_points
        if n >= N:
            choice = np.random.choice(n, N, replace=False)
            points = verts[choice]
        else:
            pad = np.zeros((N - n, 3), dtype=np.float32)
            points = np.concatenate([verts, pad], axis=0)

        return torch.from_numpy(points).float(), label  # (N, 3)


# ---------------------------------------------------------------------------
# 2. Voxel dataset (for 3D CNN)
# ---------------------------------------------------------------------------
class ShapeNetVoxel(Dataset):
    def __init__(self, root, classes=None, resolution=32, split_items=None,
                 variant="solid"):
        raw_items, self.class_to_idx = (
            (split_items, None) if split_items is not None
            else build_file_index(root, classes)
        )
        self.resolution = resolution
        assert variant in ("solid", "surface")
        self.variant = variant

        # Filter out any items missing both binvox variants, some ShapeNet
        # models don't ship a solid voxelization, so fall back to surface,
        # and drop the item entirely (with a warning) if neither exists.
        self.items = []
        dropped = 0
        for mesh_path, label in raw_items:
            resolved = self._resolve_binvox_path(mesh_path)
            if resolved is not None:
                self.items.append((resolved, label))
            else:
                dropped += 1
        if dropped:
            print(f"[ShapeNetVoxel] Warning: dropped {dropped}/{len(raw_items)} "
                  f"items with no .binvox file found.")

    def _resolve_binvox_path(self, mesh_path):
        primary = mesh_path.replace(".obj", f".{self.variant}.binvox")
        if os.path.exists(primary):
            return primary
        fallback_variant = "surface" if self.variant == "solid" else "solid"
        fallback = mesh_path.replace(".obj", f".{fallback_variant}.binvox")
        if os.path.exists(fallback):
            return fallback
        return None

    def __len__(self):
        return len(self.items)

    def _load_binvox(self, path):
        return read_binvox(path)

    def _resize_to_resolution(self, vox):
        """Downsample/pool a (typically 128^3) binvox grid to self.resolution^3
        using simple block max-pooling, so the CNN input size stays manageable."""
        r = self.resolution
        d, h, w = vox.shape
        if (d, h, w) == (r, r, r):
            return vox
        t = torch.from_numpy(vox).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        pooled = torch.nn.functional.adaptive_max_pool3d(t, output_size=(r, r, r))
        return pooled.squeeze(0).squeeze(0).numpy()

    def __getitem__(self, idx):
        binvox_path, label = self.items[idx]
        vox = self._load_binvox(binvox_path)
        vox = self._resize_to_resolution(vox)
        vox = torch.from_numpy(vox).unsqueeze(0).float()  # (1, R, R, R) channel dim
        return vox, label


# ---------------------------------------------------------------------------
# 3. Mesh dataset (for Graph CNN via PyTorch3D)
# ---------------------------------------------------------------------------
class ShapeNetMesh(Dataset):
    """Returns raw verts/faces, batch collation into a Meshes object happens
    in the collate_fn below since meshes have variable vertex/face counts."""
    def __init__(self, root, classes=None, split_items=None):
        self.items, self.class_to_idx = (
            (split_items, None) if split_items is not None
            else build_file_index(root, classes)
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        verts, faces, _ = load_obj(path, load_textures=False)
        verts = verts - verts.mean(0, keepdim=True)
        verts = verts / verts.norm(dim=1).max()
        return verts, faces.verts_idx, label


def mesh_collate_fn(batch):
    verts_list, faces_list, labels = zip(*batch)
    meshes = Meshes(verts=list(verts_list), faces=list(faces_list))
    return meshes, torch.tensor(labels, dtype=torch.long)