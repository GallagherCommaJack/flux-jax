import diffusers.models.embeddings as torch_embeddings
import diffusers.models.transformers.transformer_flux as torch_flux
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn

import flux_jax.embeddings as jax_embeddings
import jax
from flux_jax.embeddings import FluxPosEmbed, Timesteps
from flux_jax.embeddings import TimestepEmbedding

def assert_allclose_with_summary(jax_array, torch_array, atol=1 / 256, rtol=1 / 256):
    deltas = np.abs(jax_array - torch_array)

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
    | PyTorch implementation mean | {np.mean(torch_array):.2e} |
    | JAX implementation std      | {np.std(jax_array):.2e} |
    | PyTorch implementation std  | {np.std(torch_array):.2e} |
    """

    assert jnp.allclose(
        jax_array,
        torch_array,
        atol=atol,
        rtol=rtol,
    ), f"JAX and PyTorch implementations produce different results: {deltas_summary}"


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
            assert_allclose_with_summary(jr, tr.numpy())
    else:
        assert_allclose_with_summary(jax_result, torch_result.numpy())

    print(
        "JAX and PyTorch implementations of get_1d_rotary_pos_embed produce the same results."
    )


@pytest.mark.parametrize(
    ["theta", "axes_dim", "input_shape", "min_pos", "max_pos"],
    [
        (10000, [256, 256], (10 * 1024, 2), [0, 0], [100, 100]),
        (10000, [512, 512, 512], (10 * 1024, 3), [0, 0, 0], [1000, 1000, 1000]),
        (
            5000,
            [128, 256, 512, 1024],
            (10 * 1024, 4),
            [-50, -50, -50, -50],
            [50, 50, 50, 50],
        ),
    ],
)
def test_flux_pos_embed(theta, axes_dim, input_shape, min_pos, max_pos):
    # Create random input
    rng = jax.random.PRNGKey(0)
    ids = jax.random.uniform(
        rng,
        shape=input_shape,
        minval=jnp.array(min_pos),
        maxval=jnp.array(max_pos),
    )

    # JAX implementation
    jax_model = FluxPosEmbed(theta=theta, axes_dim=axes_dim)
    jax_cos, jax_sin = jax_model(ids)

    # PyTorch implementation
    torch_model = torch_flux.FluxPosEmbed(theta=theta, axes_dim=axes_dim)
    torch_ids = torch.from_numpy(np.array(ids))
    torch_cos, torch_sin = torch_model(torch_ids)

    # Compare results
    assert_allclose_with_summary(jax_cos, torch_cos.numpy())
    assert_allclose_with_summary(jax_sin, torch_sin.numpy())

    print("JAX and PyTorch implementations of FluxPosEmbed produce the same results.")

    # Test from_torch class method
    jax_model_from_torch = FluxPosEmbed.from_torch(torch_model)
    jax_cos_from_torch, jax_sin_from_torch = jax_model_from_torch(ids)

    assert_allclose_with_summary(jax_cos, jax_cos_from_torch)
    assert_allclose_with_summary(jax_sin, jax_sin_from_torch)

    print(
        "FluxPosEmbed.from_torch produces the same results as the original JAX implementation."
    )


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
    assert_allclose_with_summary(jax_embs, torch_embs.numpy())

    print(
        "JAX and PyTorch implementations of get_timestep_embedding produce the same results."
    )


@pytest.mark.parametrize(
    ["num_channels", "flip_sin_to_cos", "downscale_freq_shift", "scale", "timesteps"],
    [
        (256, True, 1, 1, np.linspace(0, 1000, 100)),
        (512, False, 0, 2, np.linspace(0, 10000, 200)),
        (1024, True, 0.5, 1.5, np.arange(0, 1000, 10)),
    ]
)
def test_timesteps(num_channels, flip_sin_to_cos, downscale_freq_shift, scale, timesteps):
    # JAX implementation
    jax_model = Timesteps(
        num_channels=num_channels,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale
    )
    jax_result = jax_model(jnp.array(timesteps))

    # PyTorch implementation
    torch_model = torch_embeddings.Timesteps(
        num_channels=num_channels,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale
    )
    torch_result = torch_model(torch.from_numpy(timesteps))

    # Compare results
    assert_allclose_with_summary(jax_result, torch_result.numpy())

    print("JAX and PyTorch implementations of Timesteps produce the same results.")

    # Test from_torch class method
    jax_model_from_torch = Timesteps.from_torch(torch_model)
    jax_result_from_torch = jax_model_from_torch(jnp.array(timesteps))

    assert_allclose_with_summary(jax_result, jax_result_from_torch)

    print("Timesteps.from_torch produces the same results as the original JAX implementation.")


@pytest.mark.parametrize(
    ["in_channels", "time_embed_dim", "act_fn", "out_dim", "post_act_fn", "cond_proj_dim"],
    [
        (256, 1024, "silu", None, None, None),
        (512, 2048, "mish", 1536, "gelu", None),
        (128, 512, "relu", 256, None, 64),
    ]
)
def test_timestep_embedding(in_channels, time_embed_dim, act_fn, out_dim, post_act_fn, cond_proj_dim):
    # Create PyTorch model
    torch_model = torch_embeddings.TimestepEmbedding(
        in_channels=in_channels,
        time_embed_dim=time_embed_dim,
        act_fn=act_fn,
        out_dim=out_dim,
        post_act_fn=post_act_fn,
        cond_proj_dim=cond_proj_dim
    ).eval().requires_grad_(False)

    # Create JAX model from PyTorch model
    jax_model = TimestepEmbedding.from_torch(torch_model)

    # Create random input
    rng = jax.random.PRNGKey(0)
    sample = jax.random.normal(rng, (1, in_channels))
    condition = None
    if cond_proj_dim is not None:
        condition = jax.random.normal(rng, (1, cond_proj_dim))

    # Run JAX model
    jax_output = jax_model(sample, condition)

    # Run PyTorch model
    torch_sample = torch.from_numpy(np.array(sample))
    torch_condition = torch.from_numpy(np.array(condition)) if condition is not None else None
    torch_output = torch_model(torch_sample, torch_condition)

    # Compare results
    assert_allclose_with_summary(jax_output, torch_output.detach().numpy())

    print("JAX and PyTorch implementations of TimestepEmbedding produce the same results.")
