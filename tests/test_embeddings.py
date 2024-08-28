
import diffusers.models.embeddings as torch_embeddings
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn

import flux_jax.embeddings as jax_embeddings


@pytest.mark.parametrize(
    ["embedding_dim", "max_period"],
    [
        (128, 100),
        (128, 1_000),
        (128, 10_000),
        (256, 10_000),
        (512, 10_000),
    ],
)
def test_get_timestep_embedding(embedding_dim, max_period):
    # Define test parameters
    timesteps = np.linspace(0, max_period, num=1000)

    # Get embeddings using JAX implementation
    jax_embs = jax_embeddings.get_timestep_embedding(
        timesteps, embedding_dim, max_period
    )

    # Get embeddings using PyTorch implementation
    torch_timesteps = torch.from_numpy(np.array(timesteps))
    torch_embs = torch_embeddings.get_timestep_embedding(
        torch_timesteps, embedding_dim, max_period
    )

    # Convert PyTorch tensor to JAX array for comparison
    torch_embs_jax = jnp.array(torch_embs.numpy())

    deltas = np.abs(jax_embs - torch_embs_jax)
    deltas_summary = f"""
    | Metric                      | Value                    |
    |-----------------------------|--------------------------|
    | Max delta                   | {np.max(deltas):.2e} |
    | Min delta                   | {np.min(deltas):.2e} |
    | Mean delta                  | {np.mean(deltas):.2e} |
    | Median delta                | {np.median(deltas):.2e} |
    | 90th percentile delta       | {np.percentile(deltas, 90):.2e} |
    | 95th percentile delta       | {np.percentile(deltas, 95):.2e} |
    | 99th percentile delta       | {np.percentile(deltas, 99):.2e} |
    | Standard deviation          | {np.std(deltas):.2e} |
    | JAX implementation mean     | {np.mean(jax_embs):.2e} |
    | PyTorch implementation mean | {np.mean(torch_embs_jax):.2e} |
    | JAX implementation std      | {np.std(jax_embs):.2e} |
    | PyTorch implementation std  | {np.std(torch_embs_jax):.2e} |
    """

    # Compare the results
    assert jnp.allclose(
        jax_embs,
        torch_embs_jax,
        atol=1 / 256,
        rtol=1 / 256,
    ), f"JAX and PyTorch implementations of get_timestep_embedding produce different results: {deltas_summary}"

    print(
        "JAX and PyTorch implementations of get_timestep_embedding produce the same results."
    )
