from __future__ import annotations

from pathlib import Path

import medmnist
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def load_pathmnist_data(data_dir: str | Path = './data', download: bool = True, batch_size: int = 64, num_workers: int = 0):
    """Load the official PathMNIST splits from MedMNIST."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.740545, 0.532982, 0.705829],
            std=[0.123682, 0.176763, 0.124431],
        ),
    ])

    train_dataset = medmnist.PathMNIST(root=str(data_dir), split='train', transform=transform, download=download)
    val_dataset = medmnist.PathMNIST(root=str(data_dir), split='val', transform=transform, download=download)
    test_dataset = medmnist.PathMNIST(root=str(data_dir), split='test', transform=transform, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'num_classes': len(np.unique(train_dataset.labels)),
        'input_dim': (3, 28, 28),
    }


def get_pathmnist_labels(dataset):
    labels = np.asarray(dataset.labels).reshape(-1).astype(int)
    return labels.tolist()
