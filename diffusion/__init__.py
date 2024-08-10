from . import gaussian_diffusion
from . import gaussian_diffusion_sample


def create_diffusion(
        # timestep_respacing,
        noise_schedule='linear',
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigma=False,
        diffusion_steps=1000,
        model=None,
        mode='train'

):
    betas = gaussian_diffusion.get_named_bata_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gaussian_diffusion.LossType.RESCALE_KL
    elif rescale_learned_sigma:
        loss_type = gaussian_diffusion.LossType.RESCALE_MSE
    else:
        loss_type = gaussian_diffusion.LossType.MSE

    return gaussian_diffusion.GaussianDiffusion(
        betas=betas,
        model_mean_type=gaussian_diffusion.ModelMeanType.EPSILON if not predict_xstart else gaussian_diffusion.ModelMeanType.START_X,
        model_var_type=(
            (
                gaussian_diffusion.ModelVarType.FIXED_LARGE if not sigma_small else gaussian_diffusion.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gaussian_diffusion.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        model=model,
        mode=mode
    )


def create_diffusion_sample(
        # timestep_respacing,
        noise_schedule='linear',
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigma=False,
        diffusion_steps=1000,
        model=None,
        apply_fn=None

):
    betas = gaussian_diffusion.get_named_bata_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gaussian_diffusion.LossType.RESCALE_KL
    elif rescale_learned_sigma:
        loss_type = gaussian_diffusion.LossType.RESCALE_MSE
    else:
        loss_type = gaussian_diffusion.LossType.MSE

    return gaussian_diffusion_sample.GaussianDiffusion(
        betas=betas,
        model_mean_type=gaussian_diffusion.ModelMeanType.EPSILON if not predict_xstart else gaussian_diffusion.ModelMeanType.START_X,
        model_var_type=(
            (
                gaussian_diffusion.ModelVarType.FIXED_LARGE if not sigma_small else gaussian_diffusion.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gaussian_diffusion.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        model=model,
        apply_fn=apply_fn
    )
