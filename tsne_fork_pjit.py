import functools

import einops
import numpy as np
import torch
from diffusers import FlaxAutoencoderKL
from flax.core import FrozenDict
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


# rng = convert_to_global_array(rng, x_sharding)

# print(x_sharding.addressable_devices)
# if jax.process_index() == 0:
#     print(x_sharding.addressable_devices)
#     print('\n' * 2)
#     print(set(mesh.devices.flat))
#
# if jax.process_index() == 0:
#     print()
#     print(rng.shape, rng.sharding.addressable_devices, )
#     print(mesh.devices)

# x = jax.device_put(jnp.ones(shape), x_sharding)

# test_sharding_jit = jax.jit(test_sharding, in_shardings= x_sharding, out_shardings=x_sharding)

def t_print(p, x):
    print(p)


def test_sharding(rng, params,vae_params, diffusion_sample, vae, shape, class_label: int, cfg_scale: float = 1.5):
    new_rng, local_rng, sample_rng = jax.random.split(rng[0], 3)

    class_labels = jnp.ones((shape[0],), dtype=jnp.int32) * class_label
    z = jax.random.normal(key=local_rng, shape=shape)
    z = jnp.concat([z, z], axis=0)
    y = jnp.array(class_labels)
    y_null = jnp.array([1000] * shape[0])
    y = jnp.concat([y, y_null], axis=0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    rng = rng.at[0].set(new_rng)

    latent = diffusion_sample.ddim_sample_loop(params, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
                                               key=sample_rng, eta=0.0)
    # latent = latent / 0.18215
    # image = vae.apply({'params': vae_params}, latent, method=vae.decode).sample
    # return einops.rearrange(image, 'b c h w->b h w c')

    return rng, latent


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
    params = model.init(rng, x, t, y, train=True)['params']

    # model.apply({'params': params}, x, t, y)

    # print(params.keys())

    # jax.tree_util.tree_map_with_path(t_print, params['final_layer'])
    # print()

    model_torch = DiT_S_2_torch()

    model_torch.load_state_dict(download_model('DiT-XL-2-256x256.pt'))

    converted_jax_params = convert_torch_to_jax(model_torch.state_dict())
    return model, converted_jax_params


def test_convert():
    # jax.distributed.initialize()
    rng = jax.random.key(0)

    device_count = jax.device_count()
    mesh_shape = (device_count,)

    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    class_label = 2

    b, h, w, c = shape = 1, 32, 32, 4

    # rng = jax.random.split(rng, num=jax.local_device_count())
    rng = jax.random.split(rng, num=jax.device_count())

    x = jnp.ones(shape)

    x_sharding = mesh_sharding(PartitionSpec('data'))

    for device in x_sharding.addressable_devices:
        if jax.process_index() == 0:
            print(device, device.coords, type(device.coords))

    model, converted_jax_params = create_state()
    diffusion_sample = create_diffusion_sample(model=model, apply_fn=model.forward_with_cfg)

    converted_jax_params = jax.tree_util.tree_map(jnp.asarray, converted_jax_params)

    vae_path = 'stabilityai/sd-vae-ft-mse'
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(pretrained_model_name_or_path=vae_path, from_pt=True)

    # vae_params = jax.device_put(vae_params, jax.local_devices()[0])

    #     pass

    vae_params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.array(x)), vae_params)

    print(converted_jax_params['x_embedder']['proj']['kernel'].devices())
    print(vae_params['decoder']['conv_in']['bias'].devices())
    print(type(converted_jax_params), type(vae_params))
    # while True:

    # vae_params = FrozenDict(vae_params)

    test_sharding_jit = shard_map(
        functools.partial(test_sharding, shape=shape, class_label=class_label, diffusion_sample=diffusion_sample,
                          vae=vae),
        mesh=mesh,
        in_specs=(PartitionSpec('data'), PartitionSpec(None),
                  PartitionSpec(None)

                  ),
        out_specs=PartitionSpec('data'))

    test_sharding_jit = jax.jit(test_sharding_jit)

    for i in range(2):
        print('Here We Go!')
        rng, numbers, = test_sharding_jit(rng, converted_jax_params,vae_params )
        b, *_ = rng.shape
        per_process_batch = b // jax.process_count()
        process_idx = jax.process_index()
        local_rng = rng[per_process_batch * process_idx: per_process_batch * (process_idx + 1)]

        # if jax.process_index() == 0:
        # print(rng.shape, numbers.shape)
        # print(local_rng)
        # print(local_rng.shape)


def show_image(img, i):
    img = img / 2 + 0.5
    print(img.max(), img.min())
    # img=img[0]
    # print(img.max(), img.min())

    plt.imshow(np.array(img[0]))
    plt.show()

    os.makedirs('imgs', exist_ok=True)

    plt.savefig(f'imgs/{i}.png')
    plt.close()


def sample_fn(diffusion_sample, params, ):
    b, h, w, c = 1, 32, 32, 4
    rng = jax.random.PRNGKey(42)

    class_labels = [2]

    n = len(class_labels)

    z = jax.random.normal(key=rng, shape=(n, h, w, c))
    z = jnp.concat([z, z], axis=0)
    y = jnp.array(class_labels)
    y_null = jnp.array([1000] * n)
    y = jnp.concat([y, y_null], axis=0)
    model_kwargs = dict(y=y, cfg_scale=7.5)

    # z = jax.random.normal(key=rng, shape=(b, h, w, c))
    # model_kwargs = dict()
    # model_kwargs = dict(y=y)
    latent = diffusion_sample.ddim_sample_loop(params, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
                                               key=rng, eta=0.0)
    return latent


if __name__ == "__main__":
    # b, h, w, c = shape = 1, 32, 32, 4
    #
    # x = jnp.ones(shape)
    # t = jnp.ones((b,), dtype=jnp.int32)
    # y = jnp.ones((b,), dtype=jnp.int32)
    # rng = jax.random.PRNGKey(42)
    # model = DiT_S_2_jax(out_channels=c, labels=1000, image_size=h, condition=True)
    # params = model.init(rng, x, t, y, train=True)['params']
    # print(params.keys())
    test_convert()
    # model, model_params = test_convert()

    """
        diffusion_sample = create_diffusion_sample(model=model, apply_fn=model.forward_with_cfg)
    
        vae_path = 'stabilityai/sd-vae-ft-mse'
    
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(pretrained_model_name_or_path=vae_path, from_pt=True)
    
        vae_params = FrozenDict(vae_params)
    
    
        @jax.jit
        def vae_decode_image(latent, vae_params):
            latent = latent / 0.18215
            image = vae.apply({'params': vae_params}, latent, method=vae.decode).sample
            return einops.rearrange(image, 'b c h w->b h w c')
    
    
        latent = sample_fn(diffusion_sample, model_params, )
        img = vae_decode_image(latent, vae_params)
        show_image(img, 0)
        """
