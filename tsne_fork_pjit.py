import argparse
import functools
import glob
import threading
import time
from pathlib import Path

import PIL.Image
import einops
import numpy as np
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


def test_sharding(rng, params, vae_params, class_label: int, diffusion_sample, vae, shape, cfg_scale: float = 1.5):
    new_rng, local_rng, sample_rng, class_rng = jax.random.split(rng[0], 4)

    # class_labels = jnp.ones((shape[0],), dtype=jnp.int32) * class_label

    class_labels = jax.random.randint(class_rng, (shape[0],), 0, 999)
    print(class_labels)

    z = jax.random.normal(key=local_rng, shape=shape)
    z = jnp.concat([z, z], axis=0)
    y = jnp.array(class_labels)
    y_null = jnp.array([1000] * shape[0])
    y = jnp.concat([y, y_null], axis=0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    rng = rng.at[0].set(new_rng)

    latent = diffusion_sample.ddim_sample_loop(params, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
                                               key=sample_rng, eta=0.0)

    latent, _ = jnp.split(latent, 2, axis=0)

    latent = latent / 0.18215
    image = vae.apply({'params': vae_params}, latent, method=vae.decode).sample
    image = image / 2 + 0.5

    image = jnp.clip(image, 0, 1)
    # image=jnp.array(image,dtype=jnp.uint8)

    image = einops.rearrange(image, 'b c h w->b h w c')
    return rng, image, class_labels


def create_state():
    b, h, w, c = shape = 1, 32, 32, 4
    rng = jax.random.PRNGKey(42)
    # x = jnp.ones(shape)

    x = jax.random.normal(rng, shape)
    t = jnp.ones((b,), dtype=jnp.int32) * 999
    y = jnp.ones((b,), dtype=jnp.int32)

    b, h, w, c = 1, 32, 32, 4
    rng = jax.random.PRNGKey(2036)
    class_labels = [0]
    n = len(class_labels)
    z = jax.random.normal(key=rng, shape=(n, h, w, c))
    x = jnp.concat([z, z], axis=0)
    y = jnp.array(class_labels)
    y_null = jnp.array([1000] * n)
    y = jnp.concat([y, y_null], axis=0)

    model = DiT_S_2_jax(out_channels=c, labels=1000, image_size=h, condition=True)
    # params = model.init(rng, x, t, y, train=True)['params']

    # model.apply({'params': params}, x, t, y)

    # print(params.keys())

    # jax.tree_util.tree_map_with_path(t_print, params['final_layer'])
    # print()

    model_torch = DiT_S_2_torch()

    model_torch.load_state_dict(download_model('DiT-XL-2-256x256.pt'))

    converted_jax_params = convert_torch_to_jax(model_torch.state_dict())
    return model, converted_jax_params


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
    print(f'{threading.active_count()=}')
    # jax.distributed.initialize()
    rng = jax.random.key(args.seed)

    device_count = jax.device_count()
    mesh_shape = (device_count,)

    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    class_label = 2

    b, h, w, c = shape = args.per_device_batch, 32, 32, 4

    # rng = jax.random.split(rng, num=jax.local_device_count())
    rng = jax.random.split(rng, num=jax.device_count())

    x = jnp.ones(shape)

    x_sharding = mesh_sharding(PartitionSpec('data'))

    # for device in x_sharding.addressable_devices:
    #     if jax.process_index() == 0:
    #         print(device, device.coords, type(device.coords))

    model, converted_jax_params = create_state()
    diffusion_sample = create_diffusion_sample(model=model, apply_fn=model.forward_with_cfg)

    converted_jax_params = jax.tree_util.tree_map(jnp.asarray, converted_jax_params)

    vae_path = 'stabilityai/sd-vae-ft-mse'
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(pretrained_model_name_or_path=vae_path, from_pt=True)

    # vae_params = jax.device_put(vae_params, jax.local_devices()[0])

    #     pass

    vae_params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x)), vae_params)

    if jax.process_index() == 0:
        print(converted_jax_params['x_embedder']['proj']['kernel'].devices())
        print(vae_params['decoder']['conv_in']['bias'].devices())
        print(type(converted_jax_params), type(vae_params))

    # vae_params = FrozenDict(vae_params)

    test_sharding_jit = shard_map(
        functools.partial(test_sharding, shape=shape, diffusion_sample=diffusion_sample,
                          vae=vae),
        mesh=mesh,
        in_specs=(PartitionSpec('data'), PartitionSpec(None),
                  PartitionSpec(None), PartitionSpec()
                  ),
        out_specs=PartitionSpec('data')

    )

    test_sharding_jit = jax.jit(test_sharding_jit)

    shard_dir_path = Path('shard_path')
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'shards-%05d.tar')
    print(shard_filename)

    counter = 0
    lock = threading.Lock()

    def thread_write(images, class_labels, sink, label, send_file=False):
        images = images * 255
        images = np.asarray(images, dtype=np.uint8)
        with lock:
            nonlocal counter

            for img, cls_label in zip(images, class_labels):
                sink.write({
                    "__key__": "%010d" % counter,
                    "jpg": PIL.Image.fromarray(img),
                    "cls": int(cls_label),
                })
                counter += 1

            if jax.process_index() == 0:
                print(counter, images.shape)

            if send_file:
                sink.shard = jax.process_index() + (label + 1) * jax.process_count()
            # sink.next_stream()
            # thread_send()

    data_per_shard = args.data_per_shard
    per_process_generate_data = b * jax.local_device_count()
    assert data_per_shard % per_process_generate_data == 0
    iter_per_shard = data_per_shard // per_process_generate_data

    sink = wds.ShardWriter(
        shard_filename,
        maxcount=data_per_shard,
        maxsize=3e10,
        start_shard=jax.process_index(),
        verbose=jax.process_index() == 0
        # maxsize=shard_size,
    )

    for label in range(0, 200):

        for i in tqdm.tqdm(range(iter_per_shard), disable=not jax.process_index() == 0):
            rng, images, class_labels = test_sharding_jit(rng, converted_jax_params, vae_params, label)
            """
            batch_size, *_ = images.shape
            per_process_batch = batch_size // jax.process_count()
            process_idx = jax.process_index()
            local_rng = rng[per_process_batch * process_idx: per_process_batch * (process_idx + 1)]
            local_images = images[per_process_batch * process_idx: per_process_batch * (process_idx + 1)]
            local_class_labels = class_labels[per_process_batch * process_idx: per_process_batch * (process_idx + 1)]
            # jnp.array().devices()


            # print(local_images.devices(),)

            print(global_array_to_host_local_array(images,mesh,PartitionSpec(None)).shape)
            local_images = jax.device_get(local_images * 255)
            """
            local_images = collect_process_data(images)
            local_class_labels = collect_process_data(class_labels)

            # if jax.process_index() == 0:
            #     print(local_images.shape, images.shape)
            # print(local_rng)
            # print(local_rng.shape)
            # print(test_sharding_jit._cache_size())
            # save_image_torch(images, i)
            # print(i, iter_per_shard)

            threading.Thread(target=thread_write,
                             args=(
                                 local_images, local_class_labels, sink, label,
                                 True if i == iter_per_shard - 1 else False)).start()
        send_file(remote_path=args.output_dir)

    while threading.active_count() > 2:
        print(f'{threading.active_count()=}')
        time.sleep(1)
    sink.close()
    print('now send file')
    send_file(0, remote_path=args.output_dir)
    while threading.active_count() > 2:
        print(f'{threading.active_count()=}')
        time.sleep(1)


def show_image(img, i):
    print(img.max(), img.min())
    plt.imshow(np.array(img[0]))
    plt.show()

    os.makedirs('imgs', exist_ok=True)

    plt.savefig(f'imgs/{i}.png')
    plt.close()


def save_image_torch(img, i):
    print(img.max(), img.min())
    img = np.array(img[:32])
    img = einops.rearrange(img, 'b  h w c->b  c h w ')
    os.makedirs('imgs', exist_ok=True)
    img = torch.from_numpy(img)
    print(img.shape)
    save_image(img, f'imgs/{i}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output-dir", default="shard_path2")
    parser.add_argument("--output-dir", default="gs://shadow-center-2b/imagenet-generated-100steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-per-shard", type=int, default=2048)
    parser.add_argument("--per-device-batch", type=int, default=128)
    test_convert(parser.parse_args())
