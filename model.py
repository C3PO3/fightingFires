import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean squared distance between each two points.

    src: [B, N, C]
    dst: [B, M, C]
    return: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Gather points by index.

    points: [B, N, C]
    idx: [B, S] or [B, S, K]
    return:
        new_points: [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS)

    xyz: [B, N, 3]
    npoint: int
    return:
        centroids: [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Group local neighboring points within radius.

    xyz: [B, N, 3]
    new_xyz: [B, S, 3]
    return:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B, S, nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sample centroids and group local regions.

    xyz: [B, N, 3]
    points: [B, N, D] or None

    return:
        new_xyz: [B, S, 3]
        new_points: [B, S, nsample, 3 + D]
    """
    fps_idx = farthest_point_sample(xyz, npoint)          # [B, S]
    new_xyz = index_points(xyz, fps_idx)                  # [B, S, 3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # [B, S, nsample]

    grouped_xyz = index_points(xyz, idx)                  # [B, S, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points, idx)        # [B, S, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        """
        in_channel: feature dimension excluding xyz
        actual conv input = in_channel + 3
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: [B, N, 3]
        points: [B, N, D] or None

        return:
            new_xyz: [B, S, 3]
            new_points: [B, S, D']
        """
        new_xyz, new_points = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points
        )  # new_points: [B, S, nsample, 3 + D]

        new_points = new_points.permute(0, 3, 2, 1)  # [B, 3+D, nsample, S]

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]      # [B, D', S]
        new_points = new_points.permute(0, 2, 1)      # [B, S, D']

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Interpolate features from xyz2 (sparser) to xyz1 (denser)

        xyz1: [B, N, 3]
        xyz2: [B, S, 3]
        points1: [B, N, D1] or None
        points2: [B, S, D2]

        return:
            new_points: [B, N, D']
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)             # [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :3]
            idx = idx[:, :, :3]                             # 3 nearest neighbors

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            grouped_points = index_points(points2, idx)     # [B, N, 3, D2]
            interpolated_points = torch.sum(
                grouped_points * weight.unsqueeze(-1), dim=2
            )                                               # [B, N, D2]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)            # [B, D, N]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)            # [B, N, D']

        return new_points


class PointNet2SemSeg(nn.Module):
    def __init__(self, in_channels=3, num_classes=13):
        """
        in_channels includes xyz.
        So if input is [x, y, z], then:
        xyz = first 3 dims
        """
        super().__init__()

        extra_feat_dim = in_channels - 3

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.2, nsample=32,
            in_channel=extra_feat_dim, mlp=[32, 32, 64]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.4, nsample=32,
            in_channel=64, mlp=[64, 64, 128]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.8, nsample=32,
            in_channel=128, mlp=[128, 128, 256]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=1.2, nsample=32,
            in_channel=256, mlp=[256, 256, 512]
        )

        # Feature Propagation layers
        self.fp4 = PointNetFeaturePropagation(
            in_channel=512 + 256, mlp=[256, 256]
        )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=256 + 128, mlp=[256, 256]
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=256 + 64, mlp=[256, 128]
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + extra_feat_dim, mlp=[128, 128, 128]
        )

        # Final classifier
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        """
        x: [B, N, in_channels]
        return: [B, N, num_classes]
        """
        xyz = x[:, :, :3]                         # [B, N, 3]
        points = x[:, :, 3:] if x.shape[-1] > 3 else None

        l0_xyz = xyz
        l0_points = points

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = l0_points.permute(0, 2, 1)           # [B, 128, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)                        # [B, num_classes, N]
        x = x.permute(0, 2, 1)                   # [B, N, num_classes]

        return x