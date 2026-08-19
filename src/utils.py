from array_api.latest import Array
import array_api_compat


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

    Note
    ----
    Negative eigenvalues of ``rho`` and ``sigma`` are clipped.
    """
    xp = array_api_compat.array_namespace(rho)
    rho_sqrt = _sqrtm(rho)
    x = rho_sqrt @ sigma @ rho_sqrt
    x_diag = xp.linalg.eigvalsh(x)
    x_diag = xp.maximum(x_diag, 0)
    return (xp.sum(xp.sqrt(x_diag)))**2


def negativity(a: Array) -> float:
    """
    Return the sum of negative eigenvalues of a matrix.

    For eigenvalues ``λ1 <= ... <= λk < 0 <= λk+1 <= ...`` returns

        ``-(λ1 + ... + λk)``


    Parameters
    ----------
    a : Array
        A Hermitian matrix.

    Returns
    -------
    float
        Minus sum of negative eigenvalues.
    """
    xp = array_api_compat.array_namespace(a)
    return -xp.sum(xp.minimum(xp.linalg.eigvalsh(a), 0))


def _sqrtm(a: Array) -> Array:
    xp = array_api_compat.array_namespace(a)
    w, v = xp.linalg.eigh(a)
    return (v * xp.sqrt(xp.maximum(w, 0))) @ xp.conj(v.T)
