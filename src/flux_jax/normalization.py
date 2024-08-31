from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
import torch.nn
from diffusers.models.normalization import (
    AdaLayerNormContinuous as AdaLayerNormContinuous_torch,
)
from diffusers.models.normalization import (
    AdaLayerNormZero as AdaLayerNormZero_torch,
)
from diffusers.models.normalization import (
    AdaLayerNormZeroSingle as AdaLayerNormZeroSingle_torch,
)
from flax import nnx

from .common import (
    get_activation,
    torch_layernorm_to_jax_layernorm,
    torch_linear_to_jax_linear,
    torch_rmsnorm_to_jax_rmsnorm,
)
from .embeddings import CombinedTimestepLabelEmbeddings


class AdaLayerNormContinuous(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.linear = nnx.Linear(
            conditioning_embedding_dim, embedding_dim * 2, use_bias=bias, rngs=rngs
        )
        if norm_type == "layer_norm":
            self.norm = nnx.LayerNorm(
                num_features=embedding_dim,
                epsilon=eps,
                use_bias=bias,
                use_scale=elementwise_affine,
                rngs=rngs,
            )
        elif norm_type == "rms_norm":
            self.norm = nnx.RMSNorm(
                num_features=embedding_dim,
                epsilon=eps,
                use_scale=elementwise_affine,
                rngs=rngs,
            )
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def __call__(self, x: jax.Array, conditioning_embedding: jax.Array) -> jax.Array:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(jax.nn.silu(conditioning_embedding).astype(x.dtype))
        scale, shift = jnp.split(emb, 2, axis=-1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

    @classmethod
    def from_torch(cls, torch_model: AdaLayerNormContinuous_torch):
        out: AdaLayerNormContinuous = nnx.eval_shape(
            lambda: cls(
                embedding_dim=1,
                conditioning_embedding_dim=1,
                elementwise_affine=False,
                eps=torch_model.norm.eps,
                bias=torch_model.linear.bias is not None,
                norm_type="layer_norm"
                if isinstance(torch_model.norm, torch.nn.LayerNorm)
                else "rms_norm",
                rngs=nnx.Rngs(0),
            )
        )
        out.linear = torch_linear_to_jax_linear(torch_model.linear)
        if isinstance(torch_model.norm, torch.nn.LayerNorm):
            out.norm = torch_layernorm_to_jax_layernorm(torch_model.norm)
        elif isinstance(torch_model.norm, torch.nn.RMSNorm):
            out.norm = torch_rmsnorm_to_jax_rmsnorm(torch_model.norm)
        return out

class AdaLayerNormZero(nnx.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        norm_type: str = "layer_norm",
        bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim, rngs=rngs)
        else:
            self.emb = None

        self.linear = nnx.Linear(embedding_dim, 6 * embedding_dim, use_bias=bias, rngs=rngs)
        if norm_type == "layer_norm":
            self.norm = nnx.LayerNorm(
                num_features=embedding_dim,
                epsilon=1e-6,
                use_bias=False,
                use_scale=False,
                rngs=rngs,
            )
        elif norm_type == "fp32_layer_norm":
            raise NotImplementedError("fp32_layer_norm is not implemented in JAX")
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )

    def __call__(
        self,
        x: jax.Array,
        timestep: Optional[jax.Array] = None,
        class_labels: Optional[jax.Array] = None,
        hidden_dtype: Optional[jnp.dtype] = None,
        emb: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(jax.nn.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(emb, 6, axis=-1)
        x = self.norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nnx.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.linear = nnx.Linear(embedding_dim, 3 * embedding_dim, use_bias=bias, rngs=rngs)
        if norm_type == "layer_norm":
            self.norm = nnx.LayerNorm(
                num_features=embedding_dim,
                epsilon=1e-6,
                use_bias=False,
                use_scale=False,
                rngs=rngs,
            )
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )

    def __call__(
        self,
        x: jax.Array,
        emb: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        emb = self.linear(jax.nn.silu(emb))
        shift_msa, scale_msa, gate_msa = jnp.split(emb, 3, axis=-1)
        x = self.norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return x, gate_msa