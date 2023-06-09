from typing import Dict, Hashable, Mapping, Optional, Union

import torch
import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Transform, MapTransform
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_tensor

class NormalizeIntensityRange(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        a_min: float,
        a_max: float,
        subtrahend: float,
        divisor: float,
        dtype: DtypeLike = np.float32
    ):
        if a_min > a_max:
            raise ValueError("a_min must be lesser than a_max.")

        self.a_min = a_min
        self.a_max = a_max

        self.subtrahend = subtrahend
        self.divisor = divisor

        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        subtrahend: Optional[float] = None,
        divisor: Optional[float] = None,
        dtype: Optional[DtypeLike] = None
    ) -> NdarrayOrTensor:
        if subtrahend is None:
            subtrahend = self.subtrahend
        if divisor is None:
            divisor = self.divisor
        if dtype is None:
            dtype = self.dtype

        img = convert_to_tensor(img, track_meta=get_track_meta())

        img = clip(img, self.a_min, self.a_max)
        img = (img - subtrahend) / divisor

        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]
        return ret

class NormalizeIntensityRanged(MapTransform):
    backend = NormalizeIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        subtrahend: float,
        divisor: float,
        dtype: Optional[DtypeLike] = np.float32,
        allow_missing_keys: bool=False
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = NormalizeIntensityRange(
            a_min, a_max,
            subtrahend, divisor,
            dtype=dtype
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.t(d[key])
        return d

