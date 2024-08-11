import enum

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import torch

from diffusion.diffusion_utils import normal_kl, discretized_gaussian_log_likelihood
from diffusion.gaussian_diffusion import LossType, ModelMeanType, ModelVarType


def mean_flat(tensor):
    return jnp.mean(tensor, axis=list(range(1, len(tensor.shape))))


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = jnp.array(arr, dtype=jnp.float32)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + jnp.zeros(broadcast_shape)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        raise NotImplemented()
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError()
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_bata_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps
        )
    else:
        raise NotImplementedError()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALE_KL


class GaussianDiffusion:

    def __init__(self, betas: int,
                 model_mean_type: ModelMeanType,
                 model_var_type: ModelVarType,
                 loss_type: LossType,
                 model: nn.Module,
                 apply_fn=None,

                 ):
        self.betas = betas
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.model = model
        self.apply_fn = apply_fn

        betas = np.array(self.betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.sample_steps = 75

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod_prev[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise):
        assert noise.shape == x_start.shape

        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape

        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, params, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, frozen_out=None):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[0], x.shape[-1]
        assert t.shape == (B,)

        if frozen_out is not None:
            model_output = frozen_out
        else:
            model_output = self.model.apply({'params': params}, x, t, **model_kwargs, method=self.apply_fn)

        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, *x.shape[1:-1], C * 2)
            model_output, model_var_values = jnp.split(model_output, 2, axis=-1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = jnp.exp(model_log_variance)
        else:
            raise NotImplementedError()

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            # if clip_denoised:
            #     x = jnp.clip(x, -1, 1)
            # x = jnp.clip(x, -1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
            raise NotImplementedError()
        else:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart,
            'extra': extra
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape

        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(self, params, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, key=None):
        out = self.p_mean_variance(params, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                   model_kwargs=model_kwargs)
        noise = jax.random.normal(key, x.shape)
        nonzero_mask = (t != 0).reshape(-1, *([1] * (len(x.shape) - 1)))
        if cond_fn is not None:
            raise NotImplementedError()
        sample = out['mean'] + nonzero_mask * jnp.exp(0.5 * out["log_variance"]) * noise
        return {'sample': sample, "pred_xstart": out['pred_xstart']}

    def p_sample_loop(self, params, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None,
                      key=None):

        if noise is not None:
            img = noise
        else:
            img = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
        indices = list(range(self.num_timesteps))[::-1]

        def loop_body(step, args):
            img, indices, key = args

            key, new_key = jax.random.split(key)

            # t = jnp.ones((shape[0],), dtype=jnp.int32) * indices[i]
            t = jnp.array(indices, dtype=jnp.int32)[step]
            t = jnp.broadcast_to(t, img.shape[0])

            out = self.p_sample(params, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                model_kwargs=model_kwargs,
                                key=key)

            key = new_key
            img = out['sample']
            return img, indices, key

        img, indices, key = jax.lax.fori_loop(0, self.num_timesteps, loop_body, init_val=(img, indices, key))

        return img

    def ddim_sample(
            self,
            params,
            x,
            t,
            prev_timestep,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
            key=None
    ):
        out = self.p_mean_variance(
            params,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        # prev_timestep = t - self.num_timesteps // self.sample_steps

        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, prev_timestep, x.shape)
        sigma = (
                eta
                * jnp.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * jnp.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = jax.random.normal(key, x.shape)
        mean_pred = (
                out["pred_xstart"] * jnp.sqrt(alpha_bar_prev)
                + jnp.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        nonzero_mask = (t != 0).reshape(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise

        # sample = jax.lax.cond(prev_timestep[0] > 0, lambda: sample, lambda: out['pred_xstart'])

        return {'sample': sample, "pred_xstart": out['pred_xstart']}

    def ddim_sample_loop(self, params, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None,
                         key=None,eta=0.0):

        if noise is not None:
            img = noise
        else:
            img = jax.random.normal(jax.random.PRNGKey(0), shape=shape)

        indices = (
            np.linspace(0, self.num_timesteps - 1, self.sample_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )

        prev_timestep_indices = np.append(indices[1:], 0)

        # print(indices)
        # print()
        # print(prev_timestep_indices)
        # print()
        # print(prev_timestep_indices[:-1] - indices[1:])

        def loop_body(step, args):
            img, indices, prev_timestep_indices, key = args
            key, new_key = jax.random.split(key)
            t = jnp.array(indices, dtype=jnp.int32)[step]
            t = jnp.broadcast_to(t, img.shape[0])

            prev_timestep = jnp.array(prev_timestep_indices, dtype=jnp.int32)[step]
            prev_timestep = jnp.broadcast_to(prev_timestep, img.shape[0])

            out = self.ddim_sample(params, img, t, prev_timestep, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                   model_kwargs=model_kwargs,
                                   key=key,eta=eta)

            key = new_key
            img = out['sample']
            return img, indices, prev_timestep_indices, key

        img, indices, prev_timestep_indices, key = jax.lax.fori_loop(0, self.sample_steps, loop_body, init_val=(
        img, indices, prev_timestep_indices, key))

        return img


if __name__ == "__main__":
    shape = (1, 6, 224, 224)
    x = jnp.zeros(shape)
    y = jnp.split(x, 2, 1)
    print(len(y))
