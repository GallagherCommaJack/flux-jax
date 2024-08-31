from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn
from diffusers.models.normalization import (
    AdaLayerNormContinuous as AdaLayerNormContinuous_torch,
    AdaLayerNormZeroSingle as AdaLayerNormZeroSingle_torch,
)
from diffusers.models.normalization import AdaLayerNormZero as AdaLayerNormZero_torch

from flux_jax.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from tests.common import assert_allclose_with_summary


@pytest.mark.parametrize(
    [
        "embedding_dim",
        "conditioning_embedding_dim",
        "elementwise_affine",
        "eps",
        "bias",
        "norm_type",
    ],
    [
        (64, 128, True, 1e-5, True, "layer_norm"),
        (128, 256, False, 1e-6, False, "layer_norm"),
        (256, 512, True, 1e-5, True, "rms_norm"),
        (512, 1024, False, 1e-6, False, "rms_norm"),
    ],
)
def test_ada_layer_norm_continuous(
    embedding_dim: int,
    conditioning_embedding_dim: int,
    elementwise_affine: bool,
    eps: float,
    bias: bool,
    norm_type: str,
):
    # Create PyTorch model
    torch_model = (
        AdaLayerNormContinuous_torch(
            embedding_dim=embedding_dim,
            conditioning_embedding_dim=conditioning_embedding_dim,
            elementwise_affine=elementwise_affine,
            eps=eps,
            bias=bias,
            norm_type=norm_type,
        )
        .eval()
        .requires_grad_(False)
    )

    # Create JAX model from PyTorch model
    jax_model = AdaLayerNormContinuous.from_torch(torch_model)

    # Create random inputs
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 10, embedding_dim))
    conditioning_embedding = jax.random.normal(
        jax.random.split(rng)[0], (2, conditioning_embedding_dim)
    )

    # Run JAX model
    jax_output = jax_model(x, conditioning_embedding)

    # Run PyTorch model
    torch_x = torch.from_numpy(np.array(x))
    torch_conditioning_embedding = torch.from_numpy(np.array(conditioning_embedding))
    torch_output = torch_model(torch_x, torch_conditioning_embedding)

    # Compare results
    assert_allclose_with_summary(jax_output, torch_output.detach().numpy())

    print(
        f"JAX and PyTorch implementations of AdaLayerNormContinuous ({norm_type}) produce the same results."
    )


@pytest.mark.parametrize(
    ["embedding_dim", "num_embeddings", "norm_type", "bias"],
    [
        (64, 1000, "layer_norm", True),
        (128, None, "layer_norm", False),
        (256, 2000, "layer_norm", True),
    ],
)
def test_ada_layer_norm_zero(
    embedding_dim: int,
    num_embeddings: Optional[int],
    norm_type: str,
    bias: bool,
):
    # Create PyTorch model
    torch_model = (
        AdaLayerNormZero_torch(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            norm_type=norm_type,
            bias=bias,
        )
        .eval()
        .requires_grad_(False)
    )

    # Create JAX model from PyTorch model
    jax_model = AdaLayerNormZero.from_torch(torch_model)

    # Create random inputs
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 10, embedding_dim))

    if num_embeddings is not None:
        timestep = jax.random.randint(jax.random.split(rng)[0], (2,), 0, num_embeddings)
        class_labels = jax.random.randint(
            jax.random.split(rng)[1], (2,), 0, num_embeddings
        )
    else:
        timestep = None
        class_labels = None
        emb = jax.random.normal(jax.random.split(rng)[0], (2, embedding_dim))

    # Run JAX model
    if num_embeddings is not None:
        jax_output = jax_model(x, timestep, class_labels)
    else:
        jax_output = jax_model(x, emb=emb)

    # Run PyTorch model
    torch_x = torch.from_numpy(np.array(x))
    if num_embeddings is not None:
        torch_timestep = torch.from_numpy(np.array(timestep))
        torch_class_labels = torch.from_numpy(np.array(class_labels))
        torch_output = torch_model(torch_x, torch_timestep, torch_class_labels)
    else:
        torch_emb = torch.from_numpy(np.array(emb))
        torch_output = torch_model(torch_x, emb=torch_emb)

    # Compare results
    for jax_out, torch_out in zip(jax_output, torch_output):
        assert_allclose_with_summary(jax_out, torch_out.detach().numpy())

    print(
        f"JAX and PyTorch implementations of AdaLayerNormZero produce the same results."
    )

@pytest.mark.parametrize(
    ["embedding_dim", "norm_type", "bias"],
    [
        (64, "layer_norm", True),
        (128, "layer_norm", False),
        (256, "layer_norm", True),
    ],
)
def test_ada_layer_norm_zero_single(
    embedding_dim: int,
    norm_type: str,
    bias: bool,
):
    # Create PyTorch model
    torch_model = (
        AdaLayerNormZeroSingle_torch(
            embedding_dim=embedding_dim,
            norm_type=norm_type,
            bias=bias,
        )
        .eval()
        .requires_grad_(False)
    )

    # Create JAX model from PyTorch model
    jax_model = AdaLayerNormZeroSingle.from_torch(torch_model)

    # Create random inputs
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 10, embedding_dim))
    emb = jax.random.normal(jax.random.split(rng)[0], (2, embedding_dim))

    # Run JAX model
    jax_output = jax_model(x, emb=emb)

    # Run PyTorch model
    torch_x = torch.from_numpy(np.array(x))
    torch_emb = torch.from_numpy(np.array(emb))
    torch_output = torch_model(torch_x, emb=torch_emb)

    # Compare results
    for jax_out, torch_out in zip(jax_output, torch_output):
        assert_allclose_with_summary(jax_out, torch_out.detach().numpy())

    print(
        f"JAX and PyTorch implementations of AdaLayerNormZeroSingle produce the same results."
    )

    # Test __call__ method directly
    jax_output_direct = jax_model(x, emb=emb)
    
    for jax_out, jax_out_direct in zip(jax_output, jax_output_direct):
        assert_allclose_with_summary(jax_out, jax_out_direct)

    print(f"AdaLayerNormZeroSingle.__call__ method produces consistent results.")
