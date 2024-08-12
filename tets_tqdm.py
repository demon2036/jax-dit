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
from jax.experimental.multihost_utils import global_array_to_host_local_array, host_local_array_to_global_array


def send_file(keep_files=5):
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
        print(files)
        for file in files:
            base_name = os.path.basename(file)
            dst = 'shard_path2'
            os.makedirs(dst, exist_ok=True)
            print(base_name, files)

            def send_data_thread(src_file, dst_file):
                with wds.gopen(src_file, "rb") as fp_local:
                    data_to_write = fp_local.read()

                with wds.gopen(f'{dst_file}', "wb") as fp:
                    fp.write(data_to_write)
                    fp.flush()

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


def test_convert():
    print(f'{threading.active_count()=}')
    # jax.distributed.initialize()
    rng = jax.random.key(0)

    device_count = jax.device_count()
    mesh_shape = (device_count,)

    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    class_label = 2

    b, h, w, c = shape = 8, 32, 32, 4

    # rng = jax.random.split(rng, num=jax.local_device_count())
    rng = jax.random.split(rng, num=jax.device_count())

    x = jnp.ones(shape)

    x_sharding = mesh_sharding(PartitionSpec('data'))


    # test_sharding_jit = shard_map(
    #     functools.partial(test_sharding, shape=shape, diffusion_sample=diffusion_sample,
    #                       vae=vae),
    #     mesh=mesh,
    #     in_specs=(PartitionSpec('data'), PartitionSpec(None),
    #               PartitionSpec(None), PartitionSpec()
    #               ),
    #     out_specs=PartitionSpec('data')
    #
    # )

    host_id = jax.process_index()
    arr = host_local_array_to_global_array(np.arange(4) * host_id, mesh, PartitionSpec('data'))
    pspecs = jax.sharding.PartitionSpec('host')
    host_local_output = global_array_to_host_local_array(arr, mesh, PartitionSpec('data'))
    if jax.process_count()==0:
        print(host_local_output)





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
    test_convert()
