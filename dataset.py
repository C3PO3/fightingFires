import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    """
    Input files:
        points.npy : [N, 3]
        labels.npy : [N]

    Train mode:
        - randomly sample one block each time __getitem__ is called

    Fixed mode:
        - pre-sample all blocks once in __init__
        - __getitem__ always returns the cached block
    """
    def __init__(
        self,
        scene_paths,
        block_size=1.0,
        num_points=4096,
        min_points=100,
        num_blocks=1000,
        augment=False,
        fixed=False
    ):
        self.block_size = block_size
        self.num_points = num_points
        self.min_points = min_points
        self.num_blocks = num_blocks
        self.augment = augment
        self.fixed = fixed

        self.scenes = []

        for points_path, labels_path in scene_paths:
            points = np.load(points_path).astype(np.float32)
            labels = np.load(labels_path).astype(np.int64)

            assert len(points) == len(labels), \
                f"Number of points and labels must match for {points_path}"

            scene_info = {
                "points": points,
                "labels": labels,
                "x_min": np.min(points[:, 0]),
                "y_min": np.min(points[:, 1]),
                "x_max": np.max(points[:, 0]),
                "y_max": np.max(points[:, 1]),
            }

            self.scenes.append(scene_info)
            
        if self.fixed:
            self.cached_blocks = []
            print(f"Pre-sampling {self.num_blocks} fixed blocks...")

            for i in range(self.num_blocks):
                block_points, block_labels = self._sample_block()
                self.cached_blocks.append((block_points, block_labels))

            print("Finished pre-sampling fixed blocks.")
    
    def _print_class_distribution(self, name="Dataset"):
        all_labels = np.concatenate([scene["labels"] for scene in self.scenes])
        unique, counts = np.unique(all_labels, return_counts=True)
        total = len(all_labels)

        print(f"\n{name} Class Distribution:")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} points ({c / total * 100:.2f}%)")

    def estimate_block_distribution(self, num_samples=200):
        all_labels = []

        for _ in range(num_samples):
            _, lbls = self._sample_block()
            all_labels.append(lbls)

        all_labels = np.concatenate(all_labels)

        unique, counts = np.unique(all_labels, return_counts=True)
        total = len(all_labels)

        print("\nEstimated TRAINING distribution (after sampling):")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} ({c / total * 100:.2f}%)")

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        """
        Returns:
            block_points: FloatTensor [num_points, 3]
            block_labels: LongTensor  [num_points]
        """
        if self.fixed:
            block_points, block_labels = self.cached_blocks[idx]
        else:
            block_points, block_labels = self._sample_block()

        if self.augment:
            block_points = self._augment(block_points)

        block_points = torch.from_numpy(block_points).float()
        block_labels = torch.from_numpy(block_labels).long()

        return block_points, block_labels

    def _sample_block(self):
        # First randomly choose a scene, than randomly sample a block from the chosen scene
        for _ in range(100):
            scene = self.scenes[np.random.randint(len(self.scenes))]

            points = scene["points"]
            labels = scene["labels"]
            x_min = scene["x_min"]
            x_max = scene["x_max"]
            y_min = scene["y_min"]
            y_max = scene["y_max"]

            if x_max - x_min < self.block_size or y_max - y_min < self.block_size:
                continue

            x_start = np.random.uniform(x_min, x_max - self.block_size)
            y_start = np.random.uniform(y_min, y_max - self.block_size)

            x_end = x_start + self.block_size
            y_end = y_start + self.block_size

            mask = (
                (points[:, 0] >= x_start) & (points[:, 0] < x_end) &
                (points[:, 1] >= y_start) & (points[:, 1] < y_end)
            )

            block_pts = points[mask]
            block_lbls = labels[mask]

            if len(block_pts) < self.min_points:
                continue

            if len(block_pts) >= self.num_points:
                choice = np.random.choice(len(block_pts), self.num_points, replace=False)
            else:
                choice = np.random.choice(len(block_pts), self.num_points, replace=True)

            block_pts = block_pts[choice]
            block_lbls = block_lbls[choice]

            block_features = block_pts

            return block_features.astype(np.float32), block_lbls.astype(np.int64)

        raise RuntimeError(
            "Failed to sample a valid block after 100 attempts. "
            "Try reducing min_points or increasing block_size."
        )
    

    def _augment(self, points):
        """
        Simple augmentation on xyz only.
        """
        # random rotation around z-axis
        # theta = np.random.uniform(0, 2 * np.pi)
        # cosval = np.cos(theta)
        # sinval = np.sin(theta)
        # rotation_matrix = np.array([
        #     [cosval, -sinval, 0],
        #     [sinval,  cosval, 0],
        #     [0,       0,      1]
        # ], dtype=np.float32)

        # points = points @ rotation_matrix.T

        # # small jitter
        # jitter = np.random.normal(0, 0.01, size=points.shape).astype(np.float32)
        # points = points + jitter

        return points