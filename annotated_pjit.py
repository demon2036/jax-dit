import argparse
import functools
import glob
import threading
import time
from pathlib import Path

import PIL.Image
import einops
import numpy as np
import timm
import torch
import tqdm
from diffusers import FlaxAutoencoderKL
from flax.core import FrozenDict
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from convert_torch_to_jax import convert_torch_to_jax
from diffusion import create_diffusion_sample
from model_vit.convert_pytorch_to_flax import convert_pytorch_to_flax_vit
from model_vit.modeling import ViT
from ref.download import download_model
from ref.model_dit_torch import DiT_XL_2 as DiT_S_2_torch
from models import DiT_XL_2 as DiT_S_2_jax
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import os
import matplotlib.pyplot as plt
from prefetch import convert_to_global_array
from torchvision.utils import save_image
import webdataset as wds
from jax.experimental.multihost_utils import global_array_to_host_local_array

from test_dataset import create_dataloaders

counter = 0
shard_idx = 0
lock = threading.Lock()


# def thread_write(images, class_labels, logits, sink, label, send_file=False):
#     global counter
#     images = einops.rearrange(images, ' b c h w ->b h w c')
#     images = np.asarray(images, dtype=np.uint8)
#
#     for img, cls_label, logit in zip(images, class_labels, logits):
#         sink.write({
#             "__key__": "%010d" % counter,
#             "jpg": PIL.Image.fromarray(img),
#             "cls": int(cls_label),
#             "logits.npy": np.array(logit, dtype=np.float16)
#         })
#         counter += 1
#
#     if jax.process_index() == 0:
#         print(counter, images.shape)
#
#     if send_file:
#         sink.shard = jax.process_index() + (label + 1) * jax.process_count()


# def thread_write(images, class_labels, logits, sink, data_per_shard):
#     global counter
#     global shard_idx
#     images = einops.rearrange(images, ' b c h w ->b h w c')
#     images = np.asarray(images, dtype=np.uint8)
#
#     for img, cls_label, logit in zip(images, class_labels, logits):
#         sink.write({
#             "__key__": "%010d" % counter,
#             "jpg": PIL.Image.fromarray(img),
#             "cls": int(cls_label),
#             "logits.npy": np.array(logit, dtype=np.float16)
#         })
#         counter += 1
#
#         if counter >= data_per_shard:
#             counter = 0
#             shard_idx += 1
#             sink.shard = jax.process_index() + shard_idx * jax.process_count()
#
#     if jax.process_index() == 0:
#         print(counter, images.shape)
def thread_write(images, class_labels, logits, sink, data_per_shard):
    global counter
    global shard_idx
    global  lock

    images = einops.rearrange(images, ' b c h w ->b h w c')
    images = np.asarray(images, dtype=np.uint8)
    predict = logits.argmax(axis=1)
    correct_mask = class_labels == predict
    images = images[correct_mask]
    class_labels = class_labels[correct_mask]
    with lock:
        for img, cls_label in zip(images, class_labels):
            sink.write({
                "__key__": "%010d" % counter,
                "jpg": PIL.Image.fromarray(img),
                "cls": int(cls_label),
            })
            counter += 1

            if counter >= data_per_shard:
                counter = 0
                shard_idx += 1
                sink.shard = jax.process_index() + shard_idx * jax.process_count()

        if jax.process_index() == 0:
            print(counter, images.shape)


def send_file(keep_files=5, remote_path='shard_path2'):
    files = glob.glob('shard_path/*.tar')
    files.sort(key=lambda x: os.path.getctime(x), )

    if len(files) == 0:
        raise NotImplemented()
    elif len(files) <= keep_files:
        pass
    else:

        if keep_files == 0:
            files = files
        else:
            files = files[:-keep_files]
        # print(files)
        for file in files:
            base_name = os.path.basename(file)
            dst = remote_path
            if 'gs' not in remote_path:
                os.makedirs(dst, exist_ok=True)

            if jax.process_index() == 0:
                print(base_name, files)

            def send_data_thread(src_file, dst_file):
                with wds.gopen(src_file, "rb") as fp_local:
                    data_to_write = fp_local.read()

                with wds.gopen(f'{dst_file}', "wb") as fp:
                    fp.write(data_to_write)
                    # fp.flush()

                os.remove(src_file)

            threading.Thread(target=send_data_thread, args=(file, f'{dst}/{base_name}')).start()


def test_sharding(x, params, vit_model, ):
    x = jnp.moveaxis(x, 1, 3).astype(jnp.float32) / 0xFF
    logit_jax = vit_model.apply({'params': params['model']}, x, det=True, )
    return logit_jax


def create_state():
    model = timm.create_model('vit_large_patch14_clip_224.openai_ft_in12k_in1k', pretrained=True)
    params = convert_pytorch_to_flax_vit(model.state_dict(), model.blocks[0].attn.num_heads)
    vit_model = ViT(
        patch_size=14,
        layers=24,
        dim=1024,
        heads=16,
        # pooling="gap",
        # posemb="sincos2d"
    )

    return vit_model, params


def collect_process_data(data):
    local_data = []
    local_devices = jax.local_devices()

    for shard in data.addressable_shards:
        device = shard.device
        local_shard = shard.data
        if device in local_devices:
            # if jax.process_index() == 0:
            #     print(device, local_devices)
            local_data.append(np.array(local_shard))
    local_data = np.concatenate(local_data, axis=0)
    return local_data


def test_convert(args):
    # jax.distributed.initialize()

    device_count = jax.device_count()
    mesh_shape = (device_count,)

    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    b, h, w, c = shape = args.per_device_batch, 32, 32, 4

    model, converted_jax_params = create_state()

    converted_jax_params = jax.tree_util.tree_map(jnp.asarray, converted_jax_params)

    test_sharding_jit = shard_map(
        functools.partial(test_sharding, vit_model=model),
        mesh=mesh,
        in_specs=(PartitionSpec('data'), PartitionSpec(None)),
        out_specs=PartitionSpec('data')

    )
    x_sharding = mesh_sharding(PartitionSpec('data'))

    test_sharding_jit = jax.jit(test_sharding_jit)

    shard_dir_path = Path('shard_path')
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'shards-%05d.tar')
    print(shard_filename)

    data_per_shard = args.data_per_shard
    per_process_generate_data = b * jax.local_device_count()
    assert data_per_shard % per_process_generate_data == 0

    sink = wds.ShardWriter(
        shard_filename,
        maxcount=data_per_shard,
        maxsize=3e10,
        start_shard=jax.process_index(),
        verbose=jax.process_index() == 0
        # maxsize=shard_size,
    )

    # print(per_process_generate_data)
    # while True:
    #     pass

    # dataloader = create_dataloaders('shard_path2/imagenet-generated-100steps_shards-00781.tar',
    # valid_batch_size=64) dataloader = create_dataloaders(
    # 'gs://shadow-center-2b/imagenet-generated-100steps/imagenet-generated -50steps_shards-01599.tar',
    # valid_batch_size=64)
    dataloader = create_dataloaders(
        'gs://shadow-center-2b/imagenet-generated-100steps/shards-{00000..06399}.tar',
        valid_batch_size=per_process_generate_data)
    # dataloader = create_dataloaders('gs://shadow-center-2b/imagenet-generated-100steps/shards-00001.tar',valid_batch_size=per_process_generate_data)
    for i, (x, y) in enumerate(dataloader):
        x, y = jax.tree_util.tree_map(np.asarray, (x, y))

        y_shard = convert_to_global_array(y,x_sharding)
        x_shard = convert_to_global_array(x, x_sharding)
        logits = test_sharding_jit(x_shard, converted_jax_params, )

        x_local = collect_process_data(x_shard)
        y_local = collect_process_data(y_shard)
        logits_local = collect_process_data(logits)

        model_predict_label = np.array(logits_local).argmax(axis=1)
        print(np.sum(model_predict_label == y_local) / x.shape[0], x.shape)

        threading.Thread(target=thread_write,
                         args=(
                             x_local, y_local, logits_local, sink, data_per_shard)).start()
    #
    #     send_file(remote_path=args.output_dir)
    #
    # while threading.active_count() > 2:
    #     print(f'{threading.active_count()=}')
    #     time.sleep(1)
    # sink.close()
    # print('now send file')
    # send_file(0, remote_path=args.output_dir)
    # while threading.active_count() > 2:
    #     print(f'{threading.active_count()=}')
    #     time.sleep(1)
    """"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output-dir", default="shard_path3")
    parser.add_argument("--output-dir", default="gs://shadow-center-2b/imagenet-generated-100steps-annotated")
    # parser.add_argument("--output-dir", default="gs://shadow-center-2b/imagenet-generated-100steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-per-shard", type=int, default=2048)
    parser.add_argument("--per-device-batch", type=int, default=256)
    test_convert(parser.parse_args())
