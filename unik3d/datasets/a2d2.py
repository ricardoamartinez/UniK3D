import json
import os

import h5py
import numpy as np
import torch

from unik3d.datasets.image_dataset import ImageDataset
from unik3d.datasets.utils import DatasetFromList


class A2D2(ImageDataset):
    min_depth = 0.01
    max_depth = 120.0
    depth_scale = 256.0
    train_split = "train_clean.txt"
    intrisics_file = "intrinsics.json"
    hdf5_paths = ["a2d2.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        resize_method="hard",
        mini=1.0,
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
            **kwargs,
        )
        self.test_mode = test_mode
        self.load_dataset()

    def load_dataset(self):
        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_paths[0]),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(h5file[self.split_file])
        txt_string = txt_file.tostring().decode("ascii")[:-1]  # correct the -1
        intrinsics = np.array(h5file[self.intrisics_file]).tostring().decode("ascii")
        intrinsics = json.loads(intrinsics)
        h5file.close()
        dataset = []
        for line in txt_string.split("\n"):
            image_filename, depth_filename = line.strip().split(" ")
            intrinsics_val = torch.tensor(
                intrinsics[os.path.join(*image_filename.split("/")[:2])]
            ).squeeze()[:, :3]
            sample = [image_filename, depth_filename, intrinsics_val]
            dataset.append(sample)

        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [False] * self.num_copies
        results["quality"] = [1] * self.num_copies
        return results
