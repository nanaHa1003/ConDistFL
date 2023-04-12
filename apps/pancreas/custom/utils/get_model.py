from importlib import import_module
from typing import Dict

import torch

def get_model(config: Dict):
    module = import_module(config["path"])
    if hasattr(module, config["name"]):
        C = getattr(module, config["name"])
        return C(**config["args"])
    else:
        raise ValueError(
            f'Unable to find {config["name"]} from module {config["path"]}'
        )

