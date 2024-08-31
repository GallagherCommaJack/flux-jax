import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn

from flux_jax.normalization import AdaLayerNormContinuous
from diffusers.models.normalization import AdaLayerNormContinuous as AdaLayerNormContinuous_torch
from tests.common import assert_allclose_with_summary

@pytest.mark.parametrize(
    ["embedding_dim", "conditioning_embedding_dim", "elementwise_affine", "eps", "bias", "norm_type"],
    [
        (64, 128, True, 1e-5, True, "layer_norm"),
        (128, 256, False, 1e-6, False, "layer_norm"),
        (256, 512, True, 1e-5, True, "rms_norm"),
        (512, 1024, False, 1e-6, False, "rms_norm"),
    ],
)
def test_ada_layer_norm_continuous(embedding_dim, conditioning_embedding_dim, elementwise_affine, eps, bias, norm_type):
    # Create PyTorch model
    torch_model = AdaLayerNormContinuous_torch(
        embedding_dim=embedding_dim,
        conditioning_embedding_dim=conditioning_embedding_dim,
        elementwise_affine=elementwise_affine,
        eps=eps,
        bias=bias,
        norm_type=norm_type,
    ).eval().requires_grad_(False)

    # Create JAX model from PyTorch model
    jax_model = AdaLayerNormContinuous.from_torch(torch_model)

    # Create random inputs
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 10, embedding_dim))
    conditioning_embedding = jax.random.normal(jax.random.split(rng)[0], (2, conditioning_embedding_dim))

    # Run JAX model
    jax_output = jax_model(x, conditioning_embedding)

    # Run PyTorch model
    torch_x = torch.from_numpy(np.array(x))
    torch_conditioning_embedding = torch.from_numpy(np.array(conditioning_embedding))
    torch_output = torch_model(torch_x, torch_conditioning_embedding)

    # Compare results
    assert_allclose_with_summary(jax_output, torch_output.detach().numpy())

    print(f"JAX and PyTorch implementations of AdaLayerNormContinuous ({norm_type}) produce the same results.")

    # Test __call__ method directly
    jax_output_direct = jax_model(x, conditioning_embedding)
    assert_allclose_with_summary(jax_output, jax_output_direct)

    print(f"AdaLayerNormContinuous.__call__ method ({norm_type}) produces consistent results.")