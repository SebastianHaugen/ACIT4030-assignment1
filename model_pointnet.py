import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Mini-PointNet that predicts a k x k transform matrix (input/feature alignment)."""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):  # x: (B, k, N)
        b = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)
        iden = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(b, 1)
        x = x + iden
        return x.view(b, self.k, self.k)


class PointNetClassifier(nn.Module):
    """Simplified PointNet classification head input transform + feature transform
    + shared MLPs + global max pool + FC classifier. """

    def __init__(self, num_classes=5, feature_transform=True):
        super().__init__()
        self.feature_transform = feature_transform
        self.input_tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        if feature_transform:
            self.feature_tnet = TNet(k=64)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):  # x: (B, N, 3)
        x = x.transpose(1, 2)  # (B, 3, N)
        trans = self.input_tnet(x)
        x = torch.bmm(trans, x)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.feature_tnet(x)
            x = torch.bmm(trans_feat, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]  # global feature (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x