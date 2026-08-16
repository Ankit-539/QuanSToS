# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:47:53 2026

@author: 3029292R
"""

import numpy as np
#Generating Haar state
def generate_haar_state(n_qubits):
    dimension = 2 ** n_qubits
    real = np.random.normal(  0.0, 1.0, dimension )
    imaginary = np.random.normal(   0.0, 1.0, dimension )
    z = real + 1j * imaginary
#normalize
    psi = z / np.linalg.norm(z)
    return psi

#Testing Haar generator
for n in [1, 6, 8]:
    psi = generate_haar_state(n)
    print(        f"{n} qubits: "        f"dimension = {len(psi)}, "        f"norm = {np.vdot(psi, psi).real:.6f}"    )
#Denisty matrix
def state_to_density_matrix(psi):
    return np.outer(  psi,  psi.conj()    )
def validate_density_matrix(rho):
    trace = np.trace(rho)
    hermitian = np.allclose(   rho,      rho.conj().T  )
    purity = np.trace(     rho @ rho )
    return trace, hermitian, purity
#Normalization test
#Haar state
psi_haar = generate_haar_state(3)
rho_haar = state_to_density_matrix(psi_haar)
print("\nHAAR STATE")
print("Dimension:", len(psi_haar))
print(  "Norm:",   np.vdot(       psi_haar,        psi_haar    ).real)

print(    "Validation:",    validate_density_matrix(rho_haar))