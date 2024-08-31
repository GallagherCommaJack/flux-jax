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

from .common import get_activation, torch_linear_to_jax_linear


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
                embedding_dim, eps=eps, use_bias=bias, use_scale=elementwise_affine
            )
        elif norm_type == "rms_norm":
            self.norm = nnx.RMSNorm(
                embedding_dim, eps=eps, use_scale=elementwise_affine
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
                embedding_dim=torch_model.norm.normalized_shape[0],
                conditioning_embedding_dim=torch_model.linear.in_features,
                elementwise_affine=torch_model.norm.elementwise_affine,
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
            out.norm.scale.value = jnp.array(torch_model.norm.weight.detach().numpy())
            out.norm.bias.value = jnp.array(torch_model.norm.bias.detach().numpy())
        elif isinstance(torch_model.norm, torch.nn.RMSNorm):
            out.norm.scale.value = jnp.array(torch_model.norm.weight.detach().numpy())
        return out
