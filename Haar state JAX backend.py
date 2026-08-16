# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 12:42:21 2026

@author: 3029292R
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from array_api_compat import array_namespace

#Generating Haar state
def generate_haar_state(n_qubits, like, seed=None):
    xp = array_namespace(like)
    backend = xp.__name__.lower()
    dimension = 2 ** n_qubits
    if "numpy" in backend:
        rng = np.random.default_rng(seed)
        real = rng.normal(0.0, 1.0, dimension)
        imaginary = rng.normal(0.0, 1.0, dimension)
        real = xp.asarray(real)
        imaginary = xp.asarray(imaginary)
    elif "jax" in backend:
        key = jax.random.PRNGKey( 0 if seed is None else seed)
        key_real, key_imaginary = jax.random.split(key)
        real = jax.random.normal( key_real, shape=(dimension,) )
        imaginary = jax.random.normal(  key_imaginary, shape=(dimension,))
    else:
        raise TypeError(  f"Unsupported array backend: {xp.__name__}"  )
    z = real + 1j * imaginary
    # normalize
    psi_haar = z / xp.linalg.vector_norm(z)
    return psi_haar

#Testing Haar generator - NumPy
numpy_like = np.asarray([0.0])
for n in [15, 24]:
    psi_haar = generate_haar_state( n,  numpy_like,  seed=42 )
    print(f"{n} qubits: "f"dimension = {len(psi_haar)}, "f"norm = {np.vdot(psi_haar, psi_haar).real:.6f}")
    
#Denisty matrix
def state_to_density_matrix(psi_haar):
    xp = array_namespace(psi_haar)
    return xp.outer( psi_haar, xp.conj(psi_haar) )
def validate_density_matrix(rho):
    xp = array_namespace(rho)
    trace = xp.trace(rho).real
    hermitian = xp.all(xp.abs(rho - xp.conj(xp.transpose(rho))) < 1e-6 )
    purity = xp.trace(rho @ rho).real
    return trace, hermitian, purity

#Normalization test

#Haar state - NumPy
psi_haar = generate_haar_state( 3, numpy_like,  seed=42)
rho_haar = state_to_density_matrix( psi_haar)
print("\nNUMPY HAAR STATE")
print("Dimension:", len(psi_haar))
print(  "Norm:",  np.vdot(  psi_haar,  psi_haar ).real)
print(  "Validation:",  validate_density_matrix( rho_haar))

# JAX TEST
jax_like = jnp.asarray([0.0])
psi_haar_jax = generate_haar_state( 3, jax_like, seed=42)
rho_haar_jax = state_to_density_matrix(  psi_haar_jax)
print("\nJAX HAAR STATE")
print("Dimension:", len(psi_haar_jax))
print( "Norm:", jnp.vdot( psi_haar_jax, psi_haar_jax).real)
print( "Validation:", validate_density_matrix(rho_haar_jax ))