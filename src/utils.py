from array_api.latest import Array


def fidelity(rho: Array, sigma: Array) -> float:
    """
    Compute fidelity of states ρ (``rho``) and σ (``sigma``):
        ``F(ρ, σ) = (Tr√X)^2,``
    where ``X = √ρ σ √ρ``.

    Parameters
    ----------
    rho : (d, d) Array
        Density matrix of the first state.
    sigma : (d, d) Array
        Density matrix of the second state.

    Returns
    -------
    float
        Fidelity of ``rho`` and ``sigma``.
    """
    pass
