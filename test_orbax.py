import os
import time

import flax
import jax
import jax.numpy as jnp
import optax
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from diffusion.gaussian_diffusion import GaussianDiffusion

from models import DiT_B_8, DiT_S_8, DiT_B_4, DiT
import orbax
from orbax.checkpoint.test_utils import erase_and_create_empty
import orbax.checkpoint as ocp
import flax.training.orbax_utils


def save():
    shape = (1, 32, 32, 3)
    x = jnp.ones(shape)
    t = jnp.ones((1,))
    y = jnp.ones((1,), dtype=jnp.int32)
    rng = jax.random.PRNGKey(42)
    model = DiT_S_8()
    params = model.init(rng, x, t, y, train=True)['params']
    print(params.keys())

    tx = optax.lion(0.1)

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    state = replicate(state)

    erase_and_create_empty('/home/jtitor/PycharmProjects/jax-dit/test')
    # checkpointer = orbax.checkpoint.CheckpointManager('/home/jtitor/PycharmProjects/jax-dit/test', )
    # checkpointer.save(step=1, args=ocp.args.StandardSave(params))
    ckpt = {'model': unreplicate(state)}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('/home/jtitor/PycharmProjects/jax-dit/test/a.ckpt', ckpt, save_args=save_args)

    print(1)


def load():
    shape = (1, 32, 32, 3)
    x = jnp.ones(shape)
    t = jnp.ones((1,))
    y = jnp.ones((1,), dtype=jnp.int32)
    rng = jax.random.PRNGKey(42)
    model = DiT_S_8()
    params = model.init(rng, x, t, y, train=True)['params']

    tx = optax.lion(0.1)

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    c = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    # new_state = c.restore('/home/jtitor/PycharmProjects/jax-dit/test/a.ckpt', args=ocp.args.StandardRestore(state))
    ckpt = {'model': state}
    new_state = c.restore('/home/jtitor/PycharmProjects/jax-dit/test/a.ckpt', item=ckpt)['model']
    print(new_state)

    # opt_state, params, step = new_state['opt_state'], new_state['params'], new_state['step']
    # print('kernel:', params['x_embedder']['proj']['kernel'].shape)
    # print(type(new_state))
    # # state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    # print(type(state.opt_state), type(opt_state))
    # state = state.replace(params=params, opt_state=tuple(opt_state))
    state = new_state

    state.apply_fn({'params': state.params}, x, t, y, False)

    def grad(params):
        out = state.apply_fn({"params": params}, x, t, y, False)
        return jnp.mean(out - jnp.zeros_like(out))

    grad = jax.grad(grad)(state.params)

    state.apply_gradients(grads=grad)

    #
    # print('kernel:', new_state.params['x_embedder']['proj']['kernel'].shape)
    # print(new_state.params['x_embedder']['proj']['kernel']-state.params['x_embedder']['proj']['kernel'])
    # print(params)
    # orbax_checkpointer = orbax.checkpoint.CheckpointManager('/home/jtitor/PycharmProjects/jax-dit/test')
    # save_args = orbax_utils.save_args_from_target(params)
    # orbax_checkpointer.save(1, params, args=ocp.args.StandardSave(params))
    # print(orbax_checkpointer.restore(1, params, args=ocp.args.StandardSave(params)))


if __name__ == "__main__":
    # save()
    load()
