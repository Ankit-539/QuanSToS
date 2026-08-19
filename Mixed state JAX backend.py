# -*- coding: utf-8 -*-

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from array_api_compat import array_namespace


def matrix_transpose(matrix, xp):
    if hasattr(xp, "matrix_transpose"):
        return xp.matrix_transpose(matrix)
    return xp.transpose(matrix)


def matrix_trace(matrix, xp):
    if hasattr(xp, "trace"):
        return xp.trace(matrix)

    diagonal = xp.stack([matrix[i, i] for i in range(matrix.shape[0])])
    return xp.sum(diagonal)


def generate_mixed_state(n_qubits, like, rank=None, seed=None):

    xp = array_namespace(like)
    backend = xp.__name__.lower()
    dimension = 2**n_qubits
    if rank is None:
        rank = dimension

    shape = (dimension, rank)

    if "numpy" in backend:
        rng = np.random.default_rng(seed)
        real = xp.asarray(rng.normal(0.0, 1.0, size=shape))
        imaginary = xp.asarray(rng.normal(0.0, 1.0, size=shape))

    elif "jax" in backend:
        key = jax.random.PRNGKey(0 if seed is None else seed)
        key_real, key_imaginary = jax.random.split(key)

        real = jax.random.normal(key_real, shape=shape)
        imaginary = jax.random.normal(key_imaginary, shape=shape)

    z = real + 1j * imaginary

    rho = z @ xp.conj(matrix_transpose(z, xp))

    # normalize
    rho = rho / matrix_trace(rho, xp)

    return rho


def validate_density_matrix(rho, tol=1e-10):
    xp = array_namespace(rho)

    trace = xp.real(matrix_trace(rho, xp))
    hermitian = xp.all(xp.abs(rho - xp.conj(matrix_transpose(rho, xp))) < tol)
    eigenvalues = xp.linalg.eigvalsh(rho)
    positive_semidefinite = xp.all(eigenvalues >= -tol)
    purity = xp.real(matrix_trace(rho @ rho, xp))

    return trace, hermitian, positive_semidefinite, purity


# Testing with NumPy
numpy_like = np.asarray([0.0])

rho_numpy = generate_mixed_state(3, numpy_like, rank=None, seed=42)

print("NUMPY MIXED STATE")
print("Dimension:", rho_numpy.shape)
print("Validation:", validate_density_matrix(rho_numpy))


# Testing with JAX
jax_like = jnp.asarray([0.0])

rho_jax = generate_mixed_state(3, jax_like, rank=None, seed=42)

print("\nJAX MIXED STATE")
print("Dimension:", rho_jax.shape)
print("Validation:", validate_density_matrix(rho_jax))
