from typing import Dict

from torch.utils.data import Dataset as _Dataset
from monai.data import DataLoader

def create_data_loader(
    dataset: _Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

