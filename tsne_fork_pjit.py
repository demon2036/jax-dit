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
from ref.model_dit_torch import DiT_XL_2 as DiT_S_2_torch
from models import DiT_XL_2 as DiT_S_2_jax
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import os
import matplotlib.pyplot as plt
from prefetch import convert_to_global_array


def t_print(p, x):
    print(p)


def test_sharding(rng):
    new_rng, local_rng = jax.random.split(rng[0], 2)
    # # print(rng.shape, )
    #
    # # numbers = jax.random.uniform(local_rng, x.shape)
    #
    # rng = rng.at[0].set(new_rng)

    return rng


def test_convert():
    # jax.distributed.initialize()
    rng = jax.random.key(0)

    device_count = jax.device_count()
    mesh_shape = (device_count,)

    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    b, h, w, c = shape = 64, 32, 32, 4

    # rng = jax.random.split(rng, num=jax.local_device_count())
    rng = jax.random.split(rng, num=jax.device_count())

    x = jnp.ones(shape)

    x_sharding = mesh_sharding(PartitionSpec('data'))

    for device in x_sharding.addressable_devices:
        if jax.process_index() == 0:
            print(device, device.coords, type(device.coords))

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

    test_sharding_jit = shard_map(test_sharding, mesh=mesh, in_specs=PartitionSpec('data'),
                                  out_specs=PartitionSpec('data'), )



    # jax.config.update('jax_threefry_partitionable', False)
    # f_exe = test_sharding_jit.lower(rng, x).compile()
    # print('Communicating?', 'collective-permute' in f_exe.as_text())
    for i in range(2):
        # rng = test_sharding_jit(rng, x)
        rng = test_sharding_jit(rng)

        # if jax.process_index() == 0:
        #     print(rng)
        #     print(rng.shape)

    """
    while True:
        pass

    
    
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

    print(params.keys())

    # jax.tree_util.tree_map_with_path(t_print, params['final_layer'])
    # print()

    model_torch = DiT_S_2_torch()

    model_torch.load_state_dict(torch.load('ref/pretrained_models/DiT-XL-2-256x256.pt'))

    converted_jax_params = convert_torch_to_jax(model_torch.state_dict())
    return model, converted_jax_params

    jax_model_out = model.apply({'params': converted_jax_params}, x, t, y, method=model.test_convert, )

    x = torch.from_numpy(np.array(einops.rearrange(x, 'b h w c-> b c h w')))
    t = torch.from_numpy(np.array(t)).to(torch.long)
    y = torch.from_numpy(np.array(y)).to(torch.long)

    with torch.no_grad():
        torch_model_out = model_torch.test_convert(x, t, y)

    # print(torch_model_out)

    np_torch_model_out = torch_model_out.numpy()

    if len(np_torch_model_out.shape) > 1 and np_torch_model_out.shape[1] == 8:
        np_torch_model_out = einops.rearrange(np_torch_model_out, 'b c h w-> b h w c')

    np_jax_model_out = np.array(jax_model_out)
    print('\n' * 5)

    print(np_jax_model_out - np_torch_model_out)
    print(np.sum(np_jax_model_out - np_torch_model_out))

    # print(np_jax_model_out)
    # print(np_torch_model_out)

    # print(np_jax_model_out)
    # print()
    #
    # print(np_torch_model_out)
    """


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
