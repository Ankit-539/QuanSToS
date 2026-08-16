# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 12:19:34 2026

@author: 3029292R
"""

import numpy as np
import jax
import jax.numpy as jnp
from array_api_compat import array_namespace
# Tensor product
def tensor_product(a, b, xp):
    return xp.reshape( a[:, None] * b[None, :], (-1,))

# Generating random Qubit
def random_single_qubit(like, seed=None):
    xp = array_namespace(like)
    backend = xp.__name__.lower()
    if "numpy" in backend:
        rng = np.random.default_rng(seed)
        u = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        u = xp.asarray(u)
        phi = xp.asarray(phi)
    elif "jax" in backend:
        key = jax.random.PRNGKey( 0 if seed is None else seed)
        key_u, key_phi = jax.random.split(key)
        u = jax.random.uniform(key_u, shape=(), minval=-1.0, maxval=1.0)
        phi = jax.random.uniform( key_phi,   shape=(), minval=0.0, maxval=2.0 * np.pi )
    else:
        raise TypeError(  f"Unsupported array backend: {xp.__name__}" )
    theta = xp.arccos(u)
    psi = xp.stack([ xp.cos(theta / 2), xp.exp(1j * phi) * xp.sin(theta / 2)])
    return psi

# Generating random product state
def generate_product_state(n_qubits, like, seed=None):
    xp = array_namespace(like)
    psi_product = xp.asarray([1.0 + 0j])
    for i in range(n_qubits):
        single_qubit = random_single_qubit( like, seed=None if seed is None else seed + i )
        psi_product = tensor_product( psi_product, single_qubit,  xp  )
    return psi_product

# Density matrix
def state_to_density_matrix(state):
    xp = array_namespace(state)
    return xp.outer( state, xp.conj(state) )

# Validation
def validate_density_matrix(rho):

    xp = array_namespace(rho)

    trace = xp.trace(rho).real

    hermitian = xp.all(
        xp.abs(
            rho - xp.conj(xp.transpose(rho))
        ) < 1e-6
    )

    purity = xp.trace(rho @ rho).real

    return trace, hermitian, purity
# Testing with numpy
numpy_like = np.asarray([0.0])
psi_product = generate_product_state(3, numpy_like, seed=42)
rho_product = state_to_density_matrix(psi_product)
print("NUMPY PRODUCT STATE")
print("Dimension:", psi_product.shape)
print( "Norm:",np.vdot(  psi_product, psi_product).real)
print(  "Validation:", validate_density_matrix(rho_product))

# Testin JAX
import jax.numpy as jnp
jax_like = jnp.asarray([0.0])
psi_product_jax = generate_product_state( 3, jax_like, seed=42)
rho_product_jax = state_to_density_matrix( psi_product_jax)
print("\nJAX PRODUCT STATE")
print("Dimension:", psi_product_jax.shape)
print( "Norm:", jnp.vdot( psi_product_jax, psi_product_jax).real)
print( "Validation:", validate_density_matrix( rho_product_jax ))