import math
import torch
from typing import Callable, Optional, Union

import diffusers.models.embeddings as torch_embeddings
import diffusers.models.transformers.transformer_flux as torch_flux
import jax
import jax.numpy as jnp
import torch.nn
from flax import nnx


def torch_linear_to_jax_linear(torch_linear: torch.nn.Linear) -> nnx.Linear:
    dense: nnx.Linear = nnx.eval_shape(
        lambda: nnx.Linear(
            in_features=torch_linear.in_features,
            out_features=torch_linear.out_features,
            use_bias=torch_linear.bias is not None,
            rngs=nnx.Rngs(0),
        )
    )
    dense.kernel.value = jnp.array(torch_linear.weight.T.numpy())
    if torch_linear.bias is not None:
        dense.bias.value = jnp.array(torch_linear.bias.numpy())
    return dense


ACTIVATION_FUNCTIONS = {
    "swish": jax.nn.silu,
    "silu": jax.nn.silu,
    "mish": jax.nn.mish,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
}


def from_torch_activation(
    torch_activation: torch.nn.Module,
) -> str:
    if isinstance(torch_activation, torch.nn.SiLU):
        return "silu"
    elif isinstance(torch_activation, torch.nn.Mish):
        return "mish"
    elif isinstance(torch_activation, torch.nn.GELU):
        return "gelu"
    elif isinstance(torch_activation, torch.nn.ReLU):
        return "relu"
    else:
        raise ValueError(f"Unsupported activation function: {torch_activation}")


def get_activation(act_fn: str) -> Callable[[jax.Array], jax.Array]:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[jax.Array, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=jnp.float32,  # jnp.float32 (hunyuan, stable audio), jnp.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`jax.Array` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`jnp.float32` or `jnp.float64`, *optional*, defaults to `jnp.float32`):
            the dtype of the frequency tensor.
    Returns:
        `jax.Array`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = jnp.arange(pos)
    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (jnp.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    t = pos.astype(freqs.dtype)  # [S]
    freqs = jnp.outer(t, freqs)  # [S, D/2]
    if use_real and repeat_interleave_real:
        freqs_cos = jnp.repeat(jnp.cos(freqs), 2, axis=1).astype(jnp.float32)  # [S, D]
        freqs_sin = jnp.repeat(jnp.sin(freqs), 2, axis=1).astype(jnp.float32)  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = jnp.concatenate([jnp.cos(freqs), jnp.cos(freqs)], axis=-1).astype(
            jnp.float32
        )  # [S, D]
        freqs_sin = jnp.concatenate([jnp.sin(freqs), jnp.sin(freqs)], axis=-1).astype(
            jnp.float32
        )  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = jnp.exp(1j * freqs).astype(jnp.float32)  # complex64     # [S, D/2]
        return freqs_cis


class FluxPosEmbed(nnx.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = jnp.squeeze(ids)
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                repeat_interleave_real=True,
                use_real=True,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = jnp.concatenate(cos_out, axis=-1)
        freqs_sin = jnp.concatenate(sin_out, axis=-1)
        return freqs_cos, freqs_sin

    @classmethod
    def from_torch(cls, torch_embeddings: torch_flux.FluxPosEmbed):
        return cls(
            theta=torch_embeddings.theta,
            axes_dim=torch_embeddings.axes_dim,
        )


def get_timestep_embedding(
    timesteps: jax.Array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (jax.Array):
            a 1-D Array of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        jax.Array: an [N x dim] Array of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * jnp.arange(
        start=0, stop=half_dim, dtype=jnp.float32
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = jnp.exp(exponent)
    emb = timesteps[:, None].astype(jnp.float32) * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = jnp.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class Timesteps(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

    @classmethod
    def from_torch(cls, torch_model: torch_embeddings.Timesteps):
        return cls(
            num_channels=torch_model.num_channels,
            flip_sin_to_cos=torch_model.flip_sin_to_cos,
            downscale_freq_shift=torch_model.downscale_freq_shift,
            scale=torch_model.scale,
        )


class TimestepEmbedding(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        *,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim: Optional[int] = None,
        sample_proj_bias: bool = True,
        rngs: nnx.Rngs,
    ):
        self.linear_1 = nnx.Linear(
            in_channels,
            time_embed_dim,
            use_bias=sample_proj_bias,
            rngs=rngs,
        )

        if cond_proj_dim is not None:
            self.cond_proj = nnx.Linear(
                cond_proj_dim,
                in_channels,
                use_bias=False,
                rngs=rngs,
            )
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nnx.Linear(
            time_embed_dim,
            time_embed_dim_out,
            use_bias=sample_proj_bias,
            rngs=rngs,
        )

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def __call__(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

    @classmethod
    def from_torch(
        cls, torch_model: torch_embeddings.TimestepEmbedding
    ) -> "TimestepEmbedding":
        out: TimestepEmbedding = nnx.eval_shape(
            lambda: cls(
                in_channels=torch_model.linear_1.in_features,
                time_embed_dim=torch_model.linear_1.out_features,
                act_fn=from_torch_activation(torch_model.act),
                out_dim=torch_model.linear_2.out_features,
                post_act_fn=from_torch_activation(torch_model.post_act)
                if torch_model.post_act is not None
                else None,
                cond_proj_dim=torch_model.cond_proj.out_features
                if torch_model.cond_proj is not None
                else None,
                sample_proj_bias=torch_model.linear_1.bias is not None,
                rngs=nnx.Rngs(0),
            )
        )
        out.linear_1 = torch_linear_to_jax_linear(torch_model.linear_1)
        out.linear_2 = torch_linear_to_jax_linear(torch_model.linear_2)
        if torch_model.cond_proj is not None:
            out.cond_proj = torch_linear_to_jax_linear(torch_model.cond_proj)
        return out


class PixArtAlphaTextProjection(nnx.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(
        self,
        in_features,
        hidden_size,
        out_features=None,
        act_fn="gelu_tanh",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nnx.Linear(
            in_features=in_features, out_features=hidden_size, use_bias=True, rngs=rngs
        )
        if act_fn == "gelu_tanh":
            self.act_1 = jax.nn.gelu
        elif act_fn == "silu":
            self.act_1 = jax.nn.silu
        elif act_fn == "silu_fp32":
            self.act_1 = lambda x: jax.nn.silu(x.astype(jnp.float32)).astype(x.dtype)
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nnx.Linear(
            in_features=hidden_size, out_features=out_features, use_bias=True, rngs=rngs
        )

    def __call__(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    @classmethod
    def from_torch(cls, torch_model: torch_embeddings.PixArtAlphaTextProjection):
        out: PixArtAlphaTextProjection = nnx.eval_shape(
            lambda: cls(
                in_features=torch_model.linear_1.in_features,
                hidden_size=torch_model.linear_1.out_features,
                out_features=torch_model.linear_2.out_features,
                act_fn="gelu_tanh"
                if isinstance(torch_model.act_1, torch.nn.GELU)
                else "silu"
                if isinstance(torch_model.act_1, torch.nn.SiLU)
                else "silu_fp32"
                if isinstance(torch_model.act_1, torch_embeddings.FP32SiLU)
                else "unknown",
                rngs=nnx.Rngs(0),
            )
        )
        out.linear_1 = torch_linear_to_jax_linear(torch_model.linear_1)
        out.linear_2 = torch_linear_to_jax_linear(torch_model.linear_2)
        return out


class CombinedTimestepTextProjEmbeddings(nnx.Module):
    def __init__(
        self, embedding_dim: int, pooled_projection_dim: int, *, rngs: nnx.Rngs
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, rngs=rngs
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu", rngs=rngs
        )

    def __call__(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.astype(dtype=pooled_projection.dtype)
        )  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning

    @classmethod
    def from_torch(
        cls, torch_model: torch_embeddings.CombinedTimestepTextProjEmbeddings
    ):
        out: CombinedTimestepTextProjEmbeddings = nnx.eval_shape(
            lambda: cls(
                embedding_dim=torch_model.timestep_embedder.linear_1.out_features,
                pooled_projection_dim=torch_model.text_embedder.linear_1.in_features,
                rngs=nnx.Rngs(0),
            )
        )
        out.time_proj = Timesteps.from_torch(torch_model.time_proj)
        out.timestep_embedder = TimestepEmbedding.from_torch(
            torch_model.timestep_embedder
        )
        out.text_embedder = PixArtAlphaTextProjection.from_torch(
            torch_model.text_embedder
        )
        return out


class CombinedTimestepGuidanceTextProjEmbeddings(nnx.Module):
    def __init__(
        self, embedding_dim: int, pooled_projection_dim: int, *, rngs: nnx.Rngs
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, rngs=rngs
        )
        self.guidance_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, rngs=rngs
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu", rngs=rngs
        )

    def __call__(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.astype(pooled_projection.dtype)
        )  # (N, D)

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(
            guidance_proj.astype(pooled_projection.dtype)
        )  # (N, D)

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning

    @classmethod
    def from_torch(
        cls, torch_model: torch_embeddings.CombinedTimestepGuidanceTextProjEmbeddings
    ):
        out: CombinedTimestepGuidanceTextProjEmbeddings = nnx.eval_shape(
            lambda: cls(
                embedding_dim=torch_model.timestep_embedder.linear_1.out_features,
                pooled_projection_dim=torch_model.text_embedder.linear_1.in_features,
                rngs=nnx.Rngs(0),
            )
        )
        out.time_proj = Timesteps.from_torch(torch_model.time_proj)
        out.timestep_embedder = TimestepEmbedding.from_torch(
            torch_model.timestep_embedder
        )
        out.guidance_embedder = TimestepEmbedding.from_torch(
            torch_model.guidance_embedder
        )
        out.text_embedder = PixArtAlphaTextProjection.from_torch(
            torch_model.text_embedder
        )
        return out
