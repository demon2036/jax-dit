import math
from typing import Any
import numpy as np
import einops
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from dataclasses import dataclass, fields
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention


@dataclass
class DiTBase:
    out_channels: int = 3

    learn_sigma: bool = True

    depth: int = 28
    dim: int = 1152
    heads: int = 16
    labels: int | None = 1000
    layerscale: bool = False
    patch_size: int = 2
    image_size: int = 32
    qk_norm: bool = False
    dropout_prob: float = 0.1
    grad_ckpt: bool = False

    learn_sigma: bool = True

    dtype: Any = jnp.float16
    precision: Any = jax.lax.Precision.DEFAULT
    use_fast_variance: bool = True
    condition: bool = True

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(DiTBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2


def scale_shift_fn(x, shift, scale):
    return x * (1 + jnp.expand_dims(scale, 1)) + jnp.expand_dims(shift, 1)


class TimestepEmbedding(DiTBase, nn.Module):
    dim: int = 768
    frequency_embedding_size: int = 256

    def setup(self):
        self.mlp = nn.Sequential([
            nn.Dense(self.dim, dtype=self.dtype, precision=self.precision),
            nn.silu,
            nn.Dense(self.dim, dtype=self.dtype, precision=self.precision),

        ])

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )

        args = t[:, None] * freqs[None]
        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concat([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Identity(nn.Module):
    def __call__(self, x):
        return x


class LabelEmbedder(DiTBase, nn.Module):

    def setup(self):
        use_cfg_embedding = self.dropout_prob > 0
        self.embedding_table = nn.Embed(self.labels + use_cfg_embedding, self.dim)

    def token_drop(self, labels, force_drop_ids=None, key=None):

        if key is None:
            key = self.make_rng('class_token_drop_key')

        if force_drop_ids is None:
            drop_ids = jax.random.uniform(key, (labels.shape[0],), minval=0, maxval=1) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.labels, labels)
        return labels

    def __call__(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class PatchEmbed(DiTBase, nn.Module):
    def setup(self):
        self.proj = nn.Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID", dtype=self.dtype, precision=self.precision
        )

    def __call__(self, x):
        x = self.proj(x)
        x = einops.rearrange(x, 'b h w c-> b (h w) c')
        return x


# class Attention(DiTBase, nn.Module):
#     def setup(self):
#         self.q_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.k_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.wq = nn.DenseGeneral((self.heads, self.head_dim), dtype=self.dtype)
#         self.wk = nn.DenseGeneral((self.heads, self.head_dim), dtype=self.dtype)
#         self.wv = nn.DenseGeneral((self.heads, self.head_dim), dtype=self.dtype)
#         self.wo = nn.DenseGeneral(self.dim, axis=(-2, -1), dtype=self.dtype)
#
#     def __call__(self, x, det: bool = True):
#         z = jnp.einsum("bqhd,bkhd->bhqk", self.q_norm(self.wq(x)) / self.head_dim ** 0.5, self.k_norm(self.wk(x)))
#         z = jnp.einsum("bhqk,bkhd->bqhd", nn.softmax(z), self.wv(x))
#         return self.wo(z)


class Attention(DiTBase, nn.Module):
    def setup(self):
        self.qkv = nn.Dense(3 * self.dim, dtype=self.dtype, precision=self.precision)

        self.proj = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision)

    def __call__(self, x, det: bool = True):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).transpose(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q / self.head_dim ** 0.5
        attn = q @ k.swapaxes(-2, -1)
        attn = nn.softmax(attn, axis=-1)
        x = attn @ v

        x = x.swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


#
# B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)
#
#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(
#                 q, k, v,
#                 dropout_p=self.attn_drop.p if self.training else 0.,
#             )
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class Attention(DiTBase, nn.Module):
#     def setup(self):
#         self.q_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.k_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.wq = nn.DenseGeneral((self.heads, self.head_dim), dtype=self.dtype)
#         self.wk = nn.DenseGeneral((self.heads, self.head_dim), dtype=self.dtype)
#         self.wv = nn.DenseGeneral((self.heads, self.head_dim), dtype=self.dtype)
#         self.wo = nn.DenseGeneral(self.dim, axis=(-2, -1), dtype=self.dtype)
#
#     def __call__(self, x, det: bool = True):
#         z = jnp.einsum("bqhd,bkhd->bhqk", self.q_norm(self.wq(x)) / self.head_dim ** 0.5, self.k_norm(self.wk(x)))
#         z = jnp.einsum("bhqk,bkhd->bqhd", nn.softmax(z), self.wv(x))
#         return self.wo(z)


# class Attention(DiTBase, nn.Module):
#     def setup(self):
#         self.q_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.k_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.wq = nn.Dense(self.dim, dtype=self.dtype)
#         self.wk = nn.Dense(self.dim, dtype=self.dtype)
#         self.wv = nn.Dense(self.dim, dtype=self.dtype)
#         self.wo = nn.Dense(self.dim,  dtype=self.dtype)
#
#
#     def __call__(self, x, det: bool = True):
#         q = einops.rearrange(self.wq(x), 'b q (h d)-> b h q d',h=self.heads)
#         k = einops.rearrange(self.wk(x), 'b q (h d)-> b h q d',h=self.heads)
#         v = einops.rearrange(self.wv(x), 'b q (h d)-> b h q d',h=self.heads)
#         z = flash_attention(q, k, v,)
#         z = einops.rearrange(z, 'b h q d ->  b q (h d) ')
#
#         return self.wo(z)


class Mlp(DiTBase, nn.Module):
    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim, dtype=self.dtype, precision=self.precision)
        self.fc2 = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision)

    def __call__(self, x):
        return self.fc2((nn.gelu(self.fc1(x), approximate=False)))


class DiTBlock(DiTBase, nn.Module):

    def setup(self):
        self.norm1 = nn.LayerNorm(use_scale=False, use_bias=False, dtype=self.dtype,
                                  use_fast_variance=self.use_fast_variance)
        self.attn = Attention(**self.kwargs)
        self.norm2 = nn.LayerNorm(use_scale=False, use_bias=False, dtype=self.dtype,
                                  use_fast_variance=self.use_fast_variance)
        self.mlp = Mlp(**self.kwargs)
        self.adaLN_modulation = nn.Sequential([
            nn.silu,
            nn.Dense(6 * self.dim, dtype=self.dtype)
        ])

    def __call__(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(self.adaLN_modulation(c), 6, 1)

        x = x + jnp.expand_dims(gate_msa, 1) * self.attn(scale_shift_fn(self.norm1(x), shift_msa, scale_msa))

        x = x + jnp.expand_dims(gate_mlp, 1) * self.mlp(scale_shift_fn(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(DiTBase, nn.Module):

    def setup(self):
        self.norm_final = nn.LayerNorm(use_scale=False, use_bias=False, use_fast_variance=self.use_fast_variance)
        out_channels = self.out_channels * 2 if self.learn_sigma else self.out_channels
        self.linear = nn.Dense(self.patch_size * self.patch_size * out_channels, dtype=self.dtype,
                               precision=self.precision)
        self.adaLN_modulation = nn.Sequential([
            nn.silu,
            nn.Dense(self.dim * 2, dtype=self.dtype, precision=self.precision)
        ])

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, 1)
        x = scale_shift_fn(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(DiTBase, nn.Module):

    def setup(self):
        self.x_embedder = PatchEmbed(**self.kwargs)
        self.t_embedder = TimestepEmbedding(**self.kwargs)
        self.y_embedder = LabelEmbedder(**self.kwargs)
        self.pos_embed = jnp.expand_dims(get_2d_sincos_pos_embed(self.dim, self.num_patches[0]), 0)

        # self.pos_embed = self.param('pos_embed',
        #                             lambda _: jnp.expand_dims(get_2d_sincos_pos_embed(self.dim, self.num_patches[0]),
        #                                                       0))

        block_fn = nn.remat(DiTBlock) if self.grad_ckpt else DiTBlock
        self.blocks = [block_fn(**self.kwargs) for _ in range(self.depth)]

        self.final_layer = FinalLayer(**self.kwargs)

    def unpatchify(self, x):
        out_channels = self.out_channels * 2 if self.learn_sigma else self.out_channels
        c = out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape((x.shape[0], h * p, h * p, c))
        return imgs

    def test_convert(self, x, t, y, train=False, cfg_scale=1.5):
        # x = self.x_embedder(x) + self.pos_embed
        # t = self.t_embedder(t)
        #
        # y = self.y_embedder(y, train)  # (N, D)
        # c = t + y
        #
        # for block in self.blocks:
        #     x = block(x, c)  # (N, T, D)
        #
        # x = self.final_layer(x, c)
        # x = self.unpatchify(x)
        # return x

        # model_out = self.__call__(x, t, y)
        # return model_out

        half = x[:len(x) // 2]
        combined = jnp.concat([half, half], axis=0)
        model_out = self.__call__(combined, t, y)
        eps, rest = model_out[:, :, :, :3], model_out[:, :, :, 3:]
        cond_eps, uncond_eps = jnp.split(eps, 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = jnp.concat([half_eps, half_eps], axis=0)
        return jnp.concat([eps, rest], axis=-1)

    def __call__(self, x, t, y=None, train=False):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)

        if self.condition:
            y = self.y_embedder(y, train)
            c = t + y
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)


        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[:len(x) // 2]
        combined = jnp.concat([half, half], axis=0)
        model_out = self.__call__(combined, t, y)
        eps, rest = model_out[:, :, :, :self.out_channels], model_out[:, :, :, self.out_channels:]
        cond_eps, uncond_eps = jnp.split(eps, 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = jnp.concat([half_eps, half_eps], axis=0)
        return jnp.concat([eps, rest], axis=-1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, dim=1152, patch_size=2, heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, dim=1152, patch_size=4, heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, dim=1152, patch_size=8, heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, dim=1024, patch_size=2, heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, dim=1024, patch_size=4, heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, dim=1024, patch_size=8, heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, dim=768, patch_size=2, heads=12, **kwargs)
    # return DiT(depth=12, dim=768, patch_size=2, heads=6, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, dim=768, patch_size=4, heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, dim=768, patch_size=8, heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, dim=384, patch_size=2, heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, dim=384, patch_size=4, heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, dim=384, patch_size=8, heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2, 'DiT-XL/4': DiT_XL_4, 'DiT-XL/8': DiT_XL_8,
    'DiT-L/2': DiT_L_2, 'DiT-L/4': DiT_L_4, 'DiT-L/8': DiT_L_8,
    'DiT-B/2': DiT_B_2, 'DiT-B/4': DiT_B_4, 'DiT-B/8': DiT_B_8,
    'DiT-S/2': DiT_S_2, 'DiT-S/4': DiT_S_4, 'DiT-S/8': DiT_S_8,
}
