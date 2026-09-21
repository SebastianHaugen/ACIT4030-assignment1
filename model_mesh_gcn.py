"""
Graph CNN classifier over meshes using PyTorch3D's GraphConv layer.

PyTorch3D's Meshes object exposes .verts_packed() (N_total_verts, 3) and
.edges_packed() (E, 2), which is what GraphConv needs. Because a batch mixes
several meshes into one packed representation, we need to global-max-pool
each mesh's vertex features separately using verts_packed_to_mesh_idx().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv


class MeshGCN(nn.Module):
    def __init__(self, num_classes=5, input_dim=3, hidden_dims=(64, 128, 256)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.gconvs = nn.ModuleList(
            [GraphConv(dims[i], dims[i + 1], init="normal", directed=False)
             for i in range(len(dims) - 1)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, meshes):
        verts = meshes.verts_packed()          # (V_total, 3)
        edges = meshes.edges_packed()          # (E_total, 2)
        num_verts_per_mesh = meshes.num_verts_per_mesh()  # (B,)
        batch_size = len(meshes)

        x = verts
        for gconv in self.gconvs:
            x = F.relu(gconv(x, edges))

        # Global average pooling per mesh 
        batch_index = torch.repeat_interleave(
            torch.arange(batch_size, device=x.device), num_verts_per_mesh
        )
        feat_dim = x.shape[1]
        sum_features = torch.zeros(batch_size, feat_dim, device=x.device, dtype=x.dtype)
        sum_features.scatter_add_(0, batch_index.unsqueeze(1).expand(-1, feat_dim), x)
        avg_features = sum_features / num_verts_per_mesh.view(-1, 1).float()

        return self.classifier(avg_features)