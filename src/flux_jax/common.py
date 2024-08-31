import jax
import jax.numpy as jnp
import torch
from flax import nnx
from typing import Callable


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


def torch_layernorm_to_jax_layernorm(
    torch_layernorm: torch.nn.LayerNorm,
) -> nnx.LayerNorm:
    jax_layernorm: nnx.LayerNorm = nnx.eval_shape(
        lambda: nnx.LayerNorm(
            num_features=torch_layernorm.normalized_shape[0],
            epsilon=torch_layernorm.eps,
            use_bias=torch_layernorm.bias is not None,
            use_scale=torch_layernorm.weight is not None,
            rngs=nnx.Rngs(0),
        )
    )
    if torch_layernorm.weight is not None:
        jax_layernorm.scale.value = jnp.array(torch_layernorm.weight.detach().numpy())
    if torch_layernorm.bias is not None:
        jax_layernorm.bias.value = jnp.array(torch_layernorm.bias.detach().numpy())
    return jax_layernorm


def torch_rmsnorm_to_jax_rmsnorm(torch_rmsnorm: torch.nn.Module) -> nnx.RMSNorm:
    # Assuming torch_rmsnorm has 'weight' attribute and 'eps' parameter
    jax_rmsnorm: nnx.RMSNorm = nnx.eval_shape(
        lambda: nnx.RMSNorm(
            num_features=torch_rmsnorm.weight.shape[0],
            epsilon=torch_rmsnorm.eps,
            use_scale=True,
            rngs=nnx.Rngs(0),
        )
    )
    jax_rmsnorm.scale.value = jnp.array(torch_rmsnorm.weight.detach().numpy())
    return jax_rmsnorm


ACTIVATION_FUNCTIONS = {
    "swish": jax.nn.silu,
    "silu": jax.nn.silu,
    "mish": jax.nn.mish,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
}


def from_torch_activation(torch_activation: torch.nn.Module) -> str:
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
        Callable: Activation function.
    """
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
