from array_api.latest import Array, ArrayNamespace
import array_api_compat

def bloch_vector(rho):
    """
    Convert a single-qubit density matrix to its Bloch vector.
    """
    xp = array_api_compat.array_namespace(counts)
    X = xp.asarray([[0, 1], [1, 0]], dtype=rho.dtype)
    Y = xp.asarray([[0, -1j], [1j, 0]], dtype=rho.dtype)
    Z = xp.asarray([[1, 0], [0, -1]], dtype=rho.dtype)

    x = xp.real(xp.trace(rho @ X))
    y = xp.real(xp.trace(rho @ Y))
    z = xp.real(xp.trace(rho @ Z))

    return xp.stack([x, y, z])