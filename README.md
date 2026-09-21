# ACIT4030-assignment1
In this assignment, you will be creating classifiers for 3D models. Use the dataset below, which is a subset of the ShapeNetCore dataset with 5 classes:

ShapeNetCore.zipLinks to an external site.

Requirements

You need to implement three different types of classifiers and compare their results:

3D Convolutional neural network using voxels as input
Graph convolutional neural network with Pythorch3D using mesh as input
PointNet using point clouds as input
Write a report containing method descriptions, implementation details, results and comparisons. Page limit is 10 pages.

Relevant help pages
Installation and general pytorch3D help:

Pytorch3D installation

Pytorch3D resources

3D CNN (Voxel):

ShapeNetCore Voxel Reader

For doing 3D voxel convolutions, I recommend using Conv3d in pytorch. Then, you basically have a similar architecture as you would have for an image-based CNN in pytorch. The difference compared to a pytorch 2D CNN other than the architecture is that you must provide your own data set class (inheriting from torch.utils.data.Dataset) to read in the voxel data instead of images.

Graph CNN (mesh):

Graph convolution in pytorch3d

PointNet (point cloud):

Mesh to point cloud
