from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.data.splits import make_train_val_indices


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    batch_size: int = 64
    num_workers: int = 2
    val_size: float = 0.15
    random_state: int = 1337
    flatten_for_stage1: bool = False


class FlattenTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


def get_cifar10_transforms(flatten: bool = False) -> transforms.Compose:
    transform_list = [transforms.ToTensor()]

    if flatten:
        transform_list.append(FlattenTransform())

    return transforms.Compose(transform_list)


def load_cifar10_datasets(
    data_dir: str = "data/raw",
    val_size: float = 0.15,
    random_state: int = 1337,
    flatten_for_stage1: bool = False,
) -> Tuple[Subset, Subset, datasets.CIFAR10]:
   
    train_transform = get_cifar10_transforms(flatten=flatten_for_stage1)
    test_transform = get_cifar10_transforms(flatten=flatten_for_stage1)

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    targets = np.array(full_train_dataset.targets)

    train_idx, val_idx = make_train_val_indices(
        n_samples=len(full_train_dataset),
        val_size=val_size,
        random_state=random_state,
        stratify_labels=targets,
    )

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    return train_subset, val_subset, test_dataset


def create_dataloaders(
    config: Optional[DataConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test.
    """
    if config is None:
        config = DataConfig()

    train_dataset, val_dataset, test_dataset = load_cifar10_datasets(
        data_dir=config.data_dir,
        val_size=config.val_size,
        random_state=config.random_state,
        flatten_for_stage1=config.flatten_for_stage1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, test_loader


def dataset_to_numpy(dataset: torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:

    xs = []
    ys = []

    for x, y in dataset:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        xs.append(x)
        ys.append(y)

    X = np.stack(xs)
    y = np.array(ys)

    return X, y