from array_api.latest import Array


def get_random_product(n: int, xp: ArrayNamespace, rng) -> Array:
    """
    Generate a random product state on ``n`` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of a random state.
    """
    psi_product = xp.asarray([1.0 + 0j])
    for _ in range(n):
        u = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        u = xp.asarray(u)
        phi = xp.asarray(phi)
        theta = xp.arccos(u)
        psi = xp.stack([xp.cos(theta / 2), xp.exp(1j * phi) * xp.sin(theta / 2)])
        psi_product = xp.reshape(psi_product[:, None] * psi[None, :], (-1,))
    rho = xp.outer(psi_product, xp.conj(psi_product))
    return rho
    


def get_random_haar(n: int, xp: ArrayNamespace, rng) -> Array:
    """
    Generate a Haar-random state on ``n`` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of a random state.
    """
     dimension = 2 ** n
    real = rng.normal(0.0, 1.0, dimension)
    imaginary = rng.normal(0.0, 1.0, dimension)
    real = xp.asarray(real)
    imaginary = xp.asarray(imaginary)
    z = real + 1j * imaginary
    psi_haar = z / xp.linalg.vector_norm(z)
    rho = xp.outer(psi_haar, xp.conj(psi_haar))
    return rho
    


def get_random_mixed(n: int) -> Array:
    """
    Generate a random mixed state on ``n`` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of a random state.
    """
    pass
