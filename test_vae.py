import einops
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from flax.core import FrozenDict

from diffusion import create_diffusion

from models import DiT_B_8
from diffusers import FlaxAutoencoderKL, FlaxStableDiffusionPipeline

if __name__ == "__main__":
    vae_path = "CompVis/stable-diffusion-v1-4"
    vae_path = "runwayml/stable-diffusion-v1-5"
    vae_path = 'stabilityai/sd-vae-ft-ema'
    # vae, params = FlaxAutoencoderKL.from_pretrained(pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    #                                                 subfolder='vae', from_pt=True)

    vae, params = FlaxAutoencoderKL.from_pretrained(pretrained_model_name_or_path=vae_path, from_pt=True)

    variables = FrozenDict({'params': params, })
    image = Image.open('kafu.jpeg')
    image = image.convert('RGB')
    image = image.resize((256, 256))
    x = jnp.array(np.array(image))
    x /= 255
    x = jnp.expand_dims(x, 0)
    x = einops.rearrange(x, 'b h w c->b c h w')
    x = (x - 0.5) * 2
    # x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    print(params.keys())
    rng = jax.random.PRNGKey(222)
    rngs = {"params": rng, }

    #init_latent_dist = self.vae.apply({"params": params["vae"]}, image, method=self.vae.encode).latent_dist
    x = vae.apply(variables, x, method=vae.encode).latent_dist
    x = x.sample(key=rng)  #.transpose((0, 3, 1, 2))
    print(x.shape)
    #
    out = vae.apply(variables, x, method=vae.decode).sample

    # out = vae.apply(variables, x, )[0]  #.sample

    print(out.shape)
    out = einops.rearrange(out, 'b c h w ->b h w c')[0]

    out = out / 2 + 0.5
    # print(out.max(), out.min())

    plt.imshow(out)
    plt.show()

    # init_latent_dist = self.vae.apply({"params": params["vae"]}, image, method=self.vae.encode)
    # init_latents = init_latent_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))

    # x = vae.apply(params,x)

    # @jax.jit
    # def test(x):
    #     diffusion.training_losses(x, t, {'y': y}, rng=rng)
    #
    #     return model.apply({'params': params}, x, t, y)
    #
    #
    # out = test(x)
