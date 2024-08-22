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
from flax.jax_utils import unreplicate
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
    ema_params: Any
    ema_decay: float = 0.9999

    def split_keys(self):
        class_token_drop_key, new_class_token_drop_key = jax.random.split(self.class_token_drop_key)
        times_key, new_times_key = jax.random.split(self.times_key)
        noise_key, new_noise_key = jax.random.split(self.noise_key)

        rngs = {"class_token_drop_key": class_token_drop_key, 'times_key': times_key, 'noise_key': noise_key}
        updates = {"class_token_drop_key": new_class_token_drop_key, 'times_key': new_times_key,
                   'noise_key': new_noise_key}
        return rngs, updates

    def replicate(self):
        return flax.jax_utils.replicate(self).replace(
            class_token_drop_key=shard_prng_key(self.class_token_drop_key),
            times_key=shard_prng_key(self.times_key),
            noise_key=shard_prng_key(self.noise_key),
        )


def train():
    b, h, w, c = 64, 64, 64, 3
    transform = transforms.Compose([
        # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        # transforms.Normalize(mean=[0.5, ], std=[0.5, ], inplace=True)
    ])

    dataset = DiffusionImageDataSet(transform=transform)

    dataloader = DataLoader(dataset, b, shuffle=True, drop_last=True, num_workers=16, persistent_workers=True)

    shape = (b, h, w, c)
    x = jnp.ones(shape)
    t = jnp.ones((b,), dtype=jnp.int32)
    y = jnp.ones((b,), dtype=jnp.int32)
    rng = jax.random.PRNGKey(42)
    model = DiT_B_2(out_channels=c, labels=10, image_size=h, condition=False)
    params = model.init(rng, x, t, y, train=True)['params']
    print(params.keys())
    init_rngs = {"params": jax.random.PRNGKey(1)}
    diffusion = create_diffusion(model=model)

    tx = optax.adamw(learning_rate=1e-4, weight_decay=0)
    params = diffusion.init(rng, x, t, {'y': y}, key=rng)['params']
    state = TrainState.create(apply_fn=diffusion.apply, params=params, tx=tx, ema_params=params,
                              class_token_drop_key=jax.random.PRNGKey(1), times_key=jax.random.PRNGKey(11),
                              noise_key=jax.random.PRNGKey(2036))

    state = state.replicate()

    @partial(jax.pmap, axis_name='batch')
    def test(state, x):
        x = einops.rearrange(x, 'b c h w->b h w c')

        def loss_fn(params, t):
            terms = state.apply_fn({'params': params}, x, t, {'train': True}, key=rngs['noise_key'], rngs=rngs)
            terms = jax.tree_util.tree_map(jnp.mean, terms)
            return terms['loss'], terms

        rngs, updates = state.split_keys()
        t = jax.random.randint(rngs['times_key'], (x.shape[0],), 0, 999, dtype=jnp.int32)
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, t)

        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))

        new_ema_params = jax.tree_util.tree_map(
            lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
            state.ema_params, state.params)
        state = state.replace(ema_params=new_ema_params)

        state = state.replace(**updates)
        return state, metrics

    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    diffusion_sample = create_diffusion_sample(model=model, apply_fn=model.__call__)

    for i in range(1400):
        with tqdm.tqdm(dataloader) as pbar:
            for x in pbar:
                x = np.array(x)
                x = shard(x)

                state, metrics = test(state, x)
                metrics = jax.tree_util.tree_map(lambda x: x[0], metrics)
                pbar.set_postfix(metrics)

            if (i + 1) % 100 == 0:
                unreplicate_state = unreplicate(state)
                # ckpt = {'model': unreplicate_state}
                #
                # save_args = orbax_utils.save_args_from_target(ckpt)
                # checkpointer.save('/home/jtitor/PycharmProjects/jax-dit/temp', ckpt, save_args=save_args, force=True)

                sample_fn(diffusion_sample, unreplicate_state.ema_params['model'], i)


def sample_fn(diffusion_sample, params, i):
    b, h, w, c = 1, 32, 32, 3
    rng = jax.random.PRNGKey(444)
    z = jax.random.normal(key=rng, shape=(b, h, w, c))
    model_kwargs = dict()
    # model_kwargs = dict(y=y)
    img = diffusion_sample.p_sample_loop(params, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, key=rng)

    img = img / 2 + 0.5
    print(img.max(), img.min())
    # img=img[0]
    # print(img.max(), img.min())

    # plt.imshow(np.array(img[0]), cmap='gray')
    plt.imshow(np.array(img[0]))
    plt.show()

    os.makedirs('../imgs', exist_ok=True)

    plt.savefig(f'imgs/{i}.png')
    plt.close()

    # while True:
    #     pass


def sample():
    b, h, w, c = 1, 28, 28, 1

    shape = (b, h, w, c)
    x = jnp.ones(shape)
    y = jnp.ones((b,), dtype=jnp.int32)
    rng = jax.random.PRNGKey(42)
    model = DiT_S_2(out_channels=c, labels=10)

    class_labels = [1]

    n = len(class_labels)

    z = jax.random.normal(key=rng, shape=(n, h, w, c))
    z = jnp.concat([z, z], axis=0)
    y = jnp.array(class_labels)
    y_null = jnp.array([10] * n)
    y = jnp.concat([y, y_null], axis=0)
    model_kwargs = dict(y=y, cfg_scale=1.5)

    diffusion = create_diffusion_sample(model=model, apply_fn=model.forward_with_cfg)

    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    params = checkpointer.restore('/home/jtitor/PycharmProjects/jax-dit/temp', )['model']['ema_params']['model']

    img = diffusion.p_sample_loop(params, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, key=rng)
    print(img.shape)

    import matplotlib.pyplot as plt

    img = img / 2 + 0.5
    print(img.max(), img.min())

    plt.imshow(np.array(img[0]), cmap='gray')
    plt.show()

    #self, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, progress=False
    # state.apply_fn({'params': params}, x.shape, clip_denoised=False, model_kwargs={'y': y})


if __name__ == "__main__":
    train()
    # sample()
