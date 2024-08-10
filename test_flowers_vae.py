import glob
import os
from functools import partial
from typing import Any
import matplotlib.pyplot as plt

import einops
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from diffusers import FlaxAutoencoderKL
from flax.core import FrozenDict
from flax.jax_utils import unreplicate, replicate
from flax.training import train_state, orbax_utils
from flax.training.common_utils import shard_prng_key, shard
from jax.tree_util import tree_map_with_path
import optax
from jax.random import PRNGKey
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from diffusion import create_diffusion, create_diffusion_sample

from models import DiT_B_2, DiT_S_4, DiT_S_2

from torch.utils.data import DataLoader, Dataset

import orbax.checkpoint as ocp
from PIL import Image


class DiffusionImageDataSet(Dataset):
    def __init__(self, path='flower/jpg', transform=None):
        super().__init__()
        self.image_path = glob.glob(f"{path}/*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        image = image.convert('RGB')
        image = self.transform(image)
        return image


class TrainState(train_state.TrainState):
    class_token_drop_key: PRNGKey
    times_key: PRNGKey
    noise_key: PRNGKey
    gaussian_key: PRNGKey
    ema_params: Any
    ema_decay: float = 0.9999

    def split_keys(self):
        class_token_drop_key, new_class_token_drop_key = jax.random.split(self.class_token_drop_key)
        times_key, new_times_key = jax.random.split(self.times_key)
        noise_key, new_noise_key = jax.random.split(self.noise_key)
        gaussian_key, new_gaussian_key = jax.random.split(self.gaussian_key)

        rngs = {"class_token_drop_key": class_token_drop_key, 'times_key': times_key, 'noise_key': noise_key,
                'gaussian_key': gaussian_key}
        updates = {"class_token_drop_key": new_class_token_drop_key, 'times_key': new_times_key,
                   'noise_key': new_noise_key, 'gaussian_key': new_gaussian_key}
        return rngs, updates

    def replicate(self):
        return flax.jax_utils.replicate(self).replace(
            class_token_drop_key=shard_prng_key(self.class_token_drop_key),
            times_key=shard_prng_key(self.times_key),
            noise_key=shard_prng_key(self.noise_key),
        )


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def create_state():
    b, h, w, c = 64, 32, 32, 4

    shape = (b, h, w, c)
    x = jnp.ones(shape)
    t = jnp.ones((b,), dtype=jnp.int32)
    y = jnp.ones((b,), dtype=jnp.int32)
    rng = jax.random.PRNGKey(42)
    model = DiT_B_2(out_channels=c, labels=10, image_size=h, condition=False)
    params = model.init(rng, x, t, y, train=True)['params']
    print(params.keys())

    diffusion = create_diffusion(model=model)

    tx = optax.adamw(learning_rate=1e-4, weight_decay=0)
    params = diffusion.init(rng, x, t, {'y': y}, key=rng)['params']
    state = TrainState.create(apply_fn=diffusion.apply, params=params, tx=tx, ema_params=params,
                              class_token_drop_key=jax.random.PRNGKey(1), times_key=jax.random.PRNGKey(11),
                              gaussian_key=jax.random.PRNGKey(7),
                              noise_key=jax.random.PRNGKey(2036))

    diffusion_sample = create_diffusion_sample(model=model, apply_fn=model.__call__)

    return state, diffusion_sample


def train():
    b, h, w, c = 64, 32, 32, 4
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        # transforms.Resize((256,256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        # transforms.Normalize(mean=[0.5, ], std=[0.5, ], inplace=True)
    ])

    dataset = DiffusionImageDataSet(transform=transform)

    dataloader = DataLoader(dataset, b, shuffle=True, drop_last=True, num_workers=16, persistent_workers=True)

    state, diffusion_sample = create_state()

    state = state.replicate()

    vae_path = 'stabilityai/sd-vae-ft-ema'
    # vae, params = FlaxAutoencoderKL.from_pretrained(pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    #                                                 subfolder='vae', from_pt=True)

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(pretrained_model_name_or_path=vae_path, from_pt=True)

    vae_params = FrozenDict(vae_params)
    vae_params = replicate(vae_params)

    @partial(jax.pmap, axis_name='batch')
    def test(state, x, vae_params):
        # x = einops.rearrange(x, 'b c h w->b h w c')

        def loss_fn(params, t,x):
            x = vae.apply({'params': vae_params}, x, method=vae.encode).latent_dist
            x = x.sample(key=rngs['gaussian_key']) * 0.18215  # .transpose((0, 3, 1, 2))
            terms = state.apply_fn({'params': params}, x, t, {'train': True}, key=rngs['noise_key'], rngs=rngs)
            terms = jax.tree_util.tree_map(jnp.mean, terms)
            return terms['loss'], terms

        rngs, updates = state.split_keys()
        t = jax.random.randint(rngs['times_key'], (x.shape[0],), 0, 999, dtype=jnp.int32)
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, t,x)

        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))

        new_ema_params = jax.tree_util.tree_map(
            lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
            state.ema_params, state.params)
        state = state.replace(ema_params=new_ema_params)

        state = state.replace(**updates)
        return state, metrics

    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    @jax.jit
    def vae_decode_image(latent, vae_params):
        latent = latent / 0.18215
        image = vae.apply({'params': vae_params}, latent, method=vae.decode).sample
        return einops.rearrange(image, 'b c h w->b h w c')

    for i in range(5000):
        with tqdm.tqdm(dataloader) as pbar:
            for x in pbar:
                x = np.array(x)
                x = shard(x)

                state, metrics = test(state, x, vae_params)
                metrics = jax.tree_util.tree_map(lambda x: x[0], metrics)
                pbar.set_postfix(metrics)

            if (i + 1) % 100 == 0:
                unreplicate_state = unreplicate(state)
                # ckpt = {'model': unreplicate_state}
                #
                # save_args = orbax_utils.save_args_from_target(ckpt)
                # checkpointer.save('/home/jtitor/PycharmProjects/jax-dit/temp', ckpt, save_args=save_args, force=True)

                latent = sample_fn(diffusion_sample, unreplicate_state.ema_params['model'], )
                img = vae_decode_image(latent, unreplicate(vae_params))
                show_image(img, i)


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
    rng = jax.random.PRNGKey(444)
    z = jax.random.normal(key=rng, shape=(b, h, w, c))
    model_kwargs = dict()
    # model_kwargs = dict(y=y)
    latent = diffusion_sample.p_sample_loop(params, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, key=rng)
    return latent


if __name__ == "__main__":
    train()
    # sample()
