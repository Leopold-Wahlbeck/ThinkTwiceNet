from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split

'''Splits the dataset into training and validation sets'''

def make_train_val_indices(
    n_samples: int,
    val_size: float = 0.15,
    random_state: int = 1337,
    stratify_labels: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:

    all_indices = np.arange(n_samples)

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    return train_idx, val_idx