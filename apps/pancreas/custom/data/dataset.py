import sys
import json
from pathlib import Path
from typing import Dict, Literal
from monai.data import CacheDataset, Dataset
from monai.data.decathlon_datalist import load_decathlon_datalist

from .transforms import get_transforms

def create_dataset(
    app_root: str,
    config: Dict,
    split: str,
    mode: Literal["train", "validate", "infer"]
):
    data_root = config["data_root"]
    data_list = config["data_list"]

    num_samples = config.get("num_samples", 1)

    ds_config = config.get("dataset", {})
    use_cache_dataset = ds_config.get("use_cache_dataset", False)
    if use_cache_dataset:
        cache_num = ds_config.get("cache_num", sys.maxsize)
        cache_rate = ds_config.get("cache_rate", 1.0)
        num_workers = ds_config.get("num_workers", 1)

    data = load_decathlon_datalist(
        data_list,
        is_segmentation=True,
        data_list_key=split,
        base_dir=data_root
    )
    transforms = get_transforms(mode=mode, num_samples=num_samples)

    if use_cache_dataset:
        ds = CacheDataset(
            data,
            transforms,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
    else:
        ds = Dataset(data, transforms)

    return ds

