import jax
import jax.numpy as jnp


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + jnp.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * jnp.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = jnp.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    # log_cdf_plus = jnp.log(cdf_plus.clamp(min=1e-12))
    log_cdf_plus = jnp.log(cdf_plus.clip(min=1e-12))
    log_one_minus_cdf_min = jnp.log((1.0 - cdf_min).clip(min=1e-12))  #.clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = jnp.where(
        x < -0.999,
        log_cdf_plus,
        jnp.where(x > 0.999, log_one_minus_cdf_min, jnp.log(cdf_delta.clip(min=1e-12))),
        #th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
