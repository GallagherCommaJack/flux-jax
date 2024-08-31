import numpy as np
import jax.numpy as jnp

def assert_allclose_with_summary(jax_array, torch_array, atol=1/256, rtol=1/256):
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