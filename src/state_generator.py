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
    pass


def get_random_haar(n: int) -> Array:
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
    pass


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
