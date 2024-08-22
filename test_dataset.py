# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import torch
import torchvision.transforms.v2 as T
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])


def create_transforms(image_size,
                      test_crop_ratio
                      ) -> Compose:
    valid_transforms = [
        T.Resize(int(image_size / test_crop_ratio), interpolation=3),
        T.CenterCrop(image_size),
        T.PILToTensor(),
    ]
    return T.Compose(valid_transforms)


def create_dataloaders(
        valid_dataset_shards,
        valid_batch_size=1024,
        valid_loader_workers=40,
        image_size=224,
        test_crop_ratio=0.875,
) -> DataLoader[Any]:
    valid_transform = create_transforms(image_size,test_crop_ratio)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(valid_dataset_shards),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.decode("pil"),
        wds.to_tuple("jpg", "cls"),
        wds.map_tuple(valid_transform, torch.tensor),
    )
    valid_dataloader = DataLoader(
        dataset,
        batch_size=(batch_size := valid_batch_size // jax.process_count()),
        num_workers=valid_loader_workers,
        # collate_fn=partial(collate_and_pad, batch_size=batch_size),
        drop_last=False,
        prefetch_factor=20,
        persistent_workers=True,
    )
    return valid_dataloader
