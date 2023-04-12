from typing import Dict

import numpy as np
import torch

def load_weights(
    model: torch.nn.Module,
    weights: Dict[str, np.ndarray]
) -> torch.nn.Module:
    local_var_dict = model.state_dict()
    model_keys = weights.keys()
    for var_name in local_var_dict:
        if var_name in model_keys:
            w = weights[var_name]
            try:
                local_var_dict[var_name] = torch.as_tensor(
                    np.reshape(w, local_var_dict[var_name].shape)
                )
            except Exception as e:
                raise ValueError(
                    f"Convert weight from {var_name} failed with error {str(e)}"
                )
    model.load_state_dict(local_var_dict)
    return model

def extract_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    local_state_dict = model.state_dict()
    local_model_dict = {}
    for var_name in local_state_dict:
        try:
            local_model_dict[var_name] = local_state_dict[var_name].cpu().numpy()
        except Exception as e:
            raise ValueError(
                f"Convert weight from {var_name} failed with error: {str(e)}"
            )
    return local_model_dict

