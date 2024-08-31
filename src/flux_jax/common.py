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