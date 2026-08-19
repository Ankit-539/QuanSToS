import numpy as np


# Generating random mixed state
def generate_mixed_state(n_qubits, rank=None):
    dimension = 2**n_qubits

    if rank is None:
        rank = dimension

    real = np.random.normal(0.0, 1.0, size=(dimension, rank))
    imaginary = np.random.normal(0.0, 1.0, size=(dimension, rank))
    z = real + 1j * imaginary

    rho = z @ z.conj().T
    rho = rho / np.trace(rho)

    return rho


# Testing mixed state
for n in [1, 3, 5]:
    rho_mixed = generate_mixed_state(n)

    print(
        f"{n} qubits: "
        f"dimension = {rho_mixed.shape}, "
        f"trace = {np.trace(rho_mixed).real:.6f}, "
        f"purity = {np.trace(rho_mixed @ rho_mixed).real:.6f}"
    )


def validate_density_matrix(rho, tol=1e-10):
    trace = np.trace(rho)
    hermitian = np.allclose(rho, rho.conj().T, atol=tol)
    eigenvalues = np.linalg.eigvalsh(rho)
    positive_semidefinite = np.all(eigenvalues >= -tol)
    purity = np.trace(rho @ rho).real
    return trace, hermitian, positive_semidefinite, purity


# Testing mixed state
rho_mixed = generate_mixed_state(3, rank=None)
print("MIXED STATE")
print("Dimension:", rho_mixed.shape)
print("Validation:", validate_density_matrix(rho_mixed))

# Purity test
for rank in [1, 2, None]:
    rho_mixed = generate_mixed_state(3, rank=rank)
    rank_label = "full" if rank is None else rank
    print(f"rank = {rank_label}: purity = {np.trace(rho_mixed @ rho_mixed).real:.6f}")
