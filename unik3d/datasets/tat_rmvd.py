import json
import os
from copy import deepcopy
from typing import Any

import h5py
import numpy as np
import torch

from unik3d.datasets.image_dataset import ImageDataset
from unik3d.datasets.pipelines import AnnotationMask, KittiCrop
from unik3d.datasets.sequence_dataset import SequenceDataset
from unik3d.datasets.utils import DatasetFromList
from unik3d.utils import identity


class TATRMVD(SequenceDataset):
    min_depth = 0.001
    max_depth = 50.0
    depth_scale = 1000.0
    default_fps = 6
    test_split = "test.txt"
    train_split = "test.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["tanks_and_temples_rmvd.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        crop=None,
        augmentations_db={},
        normalize=True,
        resize_method="hard",
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth"],
        inplace_fields: list[str] = ["K", "cam2w"],
        **kwargs,
    ):
        super().__init__(
            image_shape=image_shape,
            split_file=split_file,
            test_mode=test_mode,
            benchmark=benchmark,
            normalize=normalize,
            augmentations_db=augmentations_db,
            resize_method=resize_method,
            mini=mini,
            num_frames=num_frames,
            decode_fields=decode_fields,
            inplace_fields=inplace_fields,
            **kwargs,
        )

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [False] * self.num_frames * self.num_copies
        results["si"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [2] * self.num_frames * self.num_copies
        return results
