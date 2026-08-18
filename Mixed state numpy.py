import numpy as np


def generate_mixed_state(n_qubits, rank=None, seed=None):
    """
    Generate a random n-qubit mixed state with the Wishart/Ginibre method.

    rank controls the purity:
    - rank = 1 gives a pure state.
    - larger rank gives a more mixed state.
    - rank = 2**n_qubits gives a full-rank random mixed state.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")

    dimension = 2**n_qubits
    if rank is None:
        rank = dimension
    if rank < 1 or rank > dimension:
        raise ValueError("rank must satisfy 1 <= rank <= 2**n_qubits")

    rng = np.random.default_rng(seed)
    real = rng.normal(0.0, 1.0, size=(dimension, rank))
    imaginary = rng.normal(0.0, 1.0, size=(dimension, rank))
    ginibre = real + 1j * imaginary

    rho = ginibre @ ginibre.conj().T
    rho = rho / np.trace(rho)
    return rho


def validate_density_matrix(rho, tol=1e-10):
    trace = np.trace(rho)
    hermitian = np.allclose(rho, rho.conj().T, atol=tol)
    eigenvalues = np.linalg.eigvalsh(rho)
    positive_semidefinite = np.all(eigenvalues >= -tol)
    purity = np.trace(rho @ rho).real
    return trace, hermitian, positive_semidefinite, purity


# Testing mixed state
rho_mixed = generate_mixed_state(3, rank=None, seed=42)
print("MIXED STATE")
print("Dimension:", rho_mixed.shape)
print("Validation:", validate_density_matrix(rho_mixed))

# Purity test
for rank in [1, 2, None]:
    rho_mixed = generate_mixed_state(3, rank=rank, seed=42)
    rank_label = "full" if rank is None else rank
    print(f"rank = {rank_label}: purity = {np.trace(rho_mixed @ rho_mixed).real:.6f}")
