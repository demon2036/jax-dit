import enum

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import torch

from diffusion.diffusion_utils import normal_kl, discretized_gaussian_log_likelihood


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


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALE_MSE = enum.auto()
    KL = enum.auto()
    RESCALE_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALE_KL


class GaussianDiffusion(nn.Module):
    betas: int
    model_mean_type: ModelMeanType
    model_var_type: ModelVarType
    loss_type: LossType
    model: nn.Module
    mode: str = 'train'

    def setup(self):
        betas = np.array(self.betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

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

    def p_mean_variance(self, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, frozen_out=None):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[0], x.shape[-1]
        assert t.shape == (B,)

        if frozen_out is not None:
            model_output = frozen_out
        else:
            model_output = self.model(x, t, **model_kwargs)

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
            if clip_denoised:
                x = jnp.clip(x, -1, 1)
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

    def p_sample(self, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, key=None):
        out = self.p_mean_variance(x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                   model_kwargs=model_kwargs)
        noise = jax.random.normal(key, x.shape)
        nonzero_mask = (t != 0).reshape(-1, *([1] * (len(x.shape) - 1)))
        if cond_fn is not None:
            raise NotImplementedError()
        sample = out['mean'] + nonzero_mask * jnp.exp(0.5 * out["log_variance"]) * noise
        return {'sample': sample, "pred_xstart": out['pred_xstart']}

    def p_sample_loop(self, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, key=False):
        if noise is not None:
            img = noise
        else:
            img = jax.random.normal(jax.random.PRNGKey(0), *shape)
        indices = list(range(self.num_timesteps))[::-1]

        # for i in indices:
        #     t = jnp.ones((shape[0],), dtype=jnp.int32) * i
        #     out = self.p_sample(img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        #                         key=key)
        #     img = out['sample']

        # jit_p_sample = jax.jit(self.p_sample)
        #
        # for i in indices:
        #     t = jnp.ones((shape[0],), dtype=jnp.int32) * i
        #     out = jit_p_sample(img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        #                        key=key)
        #     img = out['sample']

        def loop_body(i, args):
            img, indices = args

            t = jnp.ones((shape[0],), dtype=jnp.int32)  # * indices[i]
            out = self.p_sample(img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs,
                                key=key)
            img = out['sample']
            return img

        def cond_fn(args):
            i, _ = args
            return i < self.num_timesteps

        img = nn.while_loop(cond_fn, loop_body, carry_variables=(img, indices))
        # img = jax.lax.fori_loop(0, self.num_timesteps, loop_body, init_val=(img, indices))
        return img

        return out['sample']

    def _vb_terms_bpd(self, x_start, x_t, t, clip_denoised=True, model_kwargs=None, frozen_out=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, frozen_out=frozen_out
        )

        kl = normal_kl(
            true_mean, true_log_variance_clipped, out['mean'], out['log_variance']
        )

        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out['mean'], log_scales=0.5 * out['log_variance']
        )

        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = jnp.where((t == 0), decoder_nll, kl)
        return {'output': output, 'pred_xstart': out['pred_xstart']}

    def training_losses(self, x_start, t, model_kwargs=None, noise=None, key=None):

        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            assert key is not None
            noise = jax.random.normal(key, x_start.shape)
        x_t = self.q_sample(x_start, t, noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALE_KL:
            raise NotImplementedError()
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALE_MSE:
            model_output = self.model(x_t, t, **model_kwargs)

            if self.model_var_type in (
                    ModelVarType.LEARNED,
                    ModelVarType.LEARNED_RANGE
            ):
                B, C = x_t.shape[0], x_t.shape[-1]
                assert model_output.shape == (B, *x_t.shape[1:-1], C * 2)
                model_output, model_var_values = jnp.split(model_output, 2, axis=-1)
                frozen_out = jnp.concat([jax.lax.stop_gradient(model_output), model_var_values], axis=-1)
                terms['vb'] = self._vb_terms_bpd(
                    frozen_out=frozen_out,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False
                )['output']

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms['mse'] = mean_flat((target - model_output) ** 2)
            if 'vb' in terms:
                terms['loss'] = terms['mse'] + terms['vb']
            else:
                terms['loss'] = terms['mse']
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def __call__(self, *args, **kwargs):

        if self.mode == 'train':
            return self.training_losses(*args, **kwargs)
        else:
            return self.p_sample_loop(*args, **kwargs)


if __name__ == "__main__":
    shape = (1, 6, 224, 224)
    x = jnp.zeros(shape)
    y = jnp.split(x, 2, 1)
    print(len(y))
