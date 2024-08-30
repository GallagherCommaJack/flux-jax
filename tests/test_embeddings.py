import diffusers.models.embeddings as torch_embeddings
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn

import flux_jax.embeddings as jax_embeddings
import jax


def assert_allclose_with_summary(jax_array, torch_array, atol=1 / 256, rtol=1 / 256):
    torch_array_jax = jnp.array(torch_array.numpy())
    deltas = np.abs(jax_array - torch_array_jax)

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
    | JAX implementation mean     | {np.mean(jax_array):.2e} |
    | PyTorch implementation mean | {np.mean(torch_array_jax):.2e} |
    | JAX implementation std      | {np.std(jax_array):.2e} |
    | PyTorch implementation std  | {np.std(torch_array_jax):.2e} |
    """

    assert jnp.allclose(
        jax_array,
        torch_array_jax,
        atol=atol,
        rtol=rtol,
    ), f"JAX and PyTorch implementations produce different results: {deltas_summary}"


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

    # Compare the results using the helper function
    assert_allclose_with_summary(jax_embs, torch_embs)

    print(
        "JAX and PyTorch implementations of get_timestep_embedding produce the same results."
    )


@pytest.mark.parametrize(
    [
        "dim",
        "pos",
        "theta",
        "use_real",
        "linear_factor",
        "ntk_factor",
        "repeat_interleave_real",
        "freqs_dtype",
    ],
    [
        (128, 100, 10000.0, False, 1.0, 1.0, True, jnp.float32),
        (256, 1000, 10000.0, True, 1.0, 1.0, True, jnp.float32),
        (512, 10000, 10000.0, False, 2.0, 1.5, False, jnp.float64),
        (1024, np.arange(50), 5000.0, True, 1.5, 2.0, False, jnp.float32),
    ],
)
def test_get_1d_rotary_pos_embed(
    dim,
    pos,
    theta,
    use_real,
    linear_factor,
    ntk_factor,
    repeat_interleave_real,
    freqs_dtype,
):
    # JAX implementation
    jax_result = jax_embeddings.get_1d_rotary_pos_embed(
        dim,
        pos,
        theta,
        use_real,
        linear_factor,
        ntk_factor,
        repeat_interleave_real,
        freqs_dtype,
    )

    torch_result = torch_embeddings.get_1d_rotary_pos_embed(
        dim=dim,
        pos=pos,
        theta=theta,
        use_real=use_real,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
        freqs_dtype=torch.from_numpy(np.array(0.0, dtype=freqs_dtype)).dtype,
    )

    # Compare results
    if isinstance(jax_result, tuple):
        for jr, tr in zip(jax_result, torch_result):
            assert_allclose_with_summary(jr, tr)
    else:
        assert_allclose_with_summary(jax_result, torch_result)

    print(
        "JAX and PyTorch implementations of get_1d_rotary_pos_embed produce the same results."
    )
