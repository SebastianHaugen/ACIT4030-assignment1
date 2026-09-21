from model_voxel_cnn import Voxel3DCNN
from model_pointnet import PointNetClassifier
from model_mesh_gcn import MeshGCN

# Had this for showing how many parameters each method has
def count_params(m):
    return sum(p.numel() for p in m.parameters())

print("Voxel CNN:", count_params(Voxel3DCNN(num_classes=5, resolution=32)))
print("PointNet:", count_params(PointNetClassifier(num_classes=5)))
print("Mesh GCN:", count_params(MeshGCN(num_classes=5)))