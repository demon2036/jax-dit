import einops
import numpy as np
import torch
from diffusers import FlaxAutoencoderKL
from flax.core import FrozenDict

from diffusion import create_diffusion_sample
from ref.model_dit_torch import DiT_XL_2 as DiT_S_2_torch
from models import DiT_XL_2 as DiT_S_2_jax
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import os
import matplotlib.pyplot as plt


def t_print(p, x):
    print(p)


def convert_torch_to_jax(state_dict):
    print(state_dict.keys())

    # for k in state_dict.keys():
    #     if 'blocks.0' in k:
    #         print(k)
    # print(k)

    params = {
        'pos_embed': state_dict['pos_embed'],

        'x_embedder.proj.kernel': state_dict['x_embedder.proj.weight'].permute(2, 3, 1, 0),
        'x_embedder.proj.bias': state_dict['x_embedder.proj.bias'],

        't_embedder.mlp.layers_0.kernel': state_dict['t_embedder.mlp.0.weight'].transpose(1, 0),
        't_embedder.mlp.layers_0.bias': state_dict['t_embedder.mlp.0.bias'],
        't_embedder.mlp.layers_2.kernel': state_dict['t_embedder.mlp.2.weight'].transpose(1, 0),
        't_embedder.mlp.layers_2.bias': state_dict['t_embedder.mlp.2.bias'],

        'y_embedder.embedding_table.embedding': state_dict['y_embedder.embedding_table.weight'],

        'final_layer.adaLN_modulation.layers_1.kernel': state_dict['final_layer.adaLN_modulation.1.weight'].transpose(1,
                                                                                                                      0),
        'final_layer.adaLN_modulation.layers_1.bias': state_dict['final_layer.adaLN_modulation.1.bias'],
        'final_layer.linear.kernel': state_dict['final_layer.linear.weight'].transpose(1, 0),
        'final_layer.linear.bias': state_dict['final_layer.linear.bias'],

    }

    layer_idx = 0
    while f"blocks.{layer_idx}.attn.qkv.bias" in state_dict:
        attn_w_qkv = state_dict[f'blocks.{layer_idx}.attn.qkv.weight'].transpose(1, 0)
        attn_w_proj = state_dict[f'blocks.{layer_idx}.attn.proj.weight'].transpose(1, 0)
        attn_bias_qkv = state_dict[f'blocks.{layer_idx}.attn.qkv.bias']
        attn_bias_proj = state_dict[f'blocks.{layer_idx}.attn.proj.bias']

        params[f'blocks_{layer_idx}.attn.qkv.kernel'] = attn_w_qkv
        params[f'blocks_{layer_idx}.attn.proj.kernel'] = attn_w_proj
        params[f'blocks_{layer_idx}.attn.qkv.bias'] = attn_bias_qkv
        params[f'blocks_{layer_idx}.attn.proj.bias'] = attn_bias_proj

        mlp_w_fc1 = state_dict[f'blocks.{layer_idx}.mlp.fc1.weight'].transpose(1, 0)
        mlp_w_fc2 = state_dict[f'blocks.{layer_idx}.mlp.fc2.weight'].transpose(1, 0)
        mlp_bias_fc1 = state_dict[f'blocks.{layer_idx}.mlp.fc1.bias']
        mlp_bias_fc2 = state_dict[f'blocks.{layer_idx}.mlp.fc2.bias']

        params[f'blocks_{layer_idx}.mlp.fc1.kernel'] = mlp_w_fc1
        params[f'blocks_{layer_idx}.mlp.fc2.kernel'] = mlp_w_fc2
        params[f'blocks_{layer_idx}.mlp.fc1.bias'] = mlp_bias_fc1
        params[f'blocks_{layer_idx}.mlp.fc2.bias'] = mlp_bias_fc2

        adaLN_modulation_layers_1_kernel = state_dict[f'blocks.{layer_idx}.adaLN_modulation.1.weight'].transpose(1, 0)
        adaLN_modulation_layers_1_bias = state_dict[f'blocks.{layer_idx}.adaLN_modulation.1.bias']

        params[f'blocks_{layer_idx}.adaLN_modulation.layers_1.kernel'] = adaLN_modulation_layers_1_kernel
        params[f'blocks_{layer_idx}.adaLN_modulation.layers_1.bias'] = adaLN_modulation_layers_1_bias

        layer_idx += 1

        # print(f'{layer_idx=}')

    params = {k: v.numpy() for k, v in params.items()}

    # print()
    params = flax.traverse_util.unflatten_dict(params, '.')

    # jax.tree_util.tree_map_with_path(t_print, params)

    # print(params)

    # while True:
    #     params

    return params


def test_convert():
    b, h, w, c = shape = 1, 32, 32, 4
    rng = jax.random.PRNGKey(42)
    # x = jnp.ones(shape)

    x = jax.random.normal(rng, shape)
    t = jnp.ones((b,), dtype=jnp.int32)*999
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
    return model,converted_jax_params

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
    latent = diffusion_sample.ddim_sample_loop(params, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, key=rng,eta=0.0)
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
    # test_convert()
    model, model_params = test_convert()
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
