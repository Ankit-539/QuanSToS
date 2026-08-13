from array_api.latest import Array
import array_api_compat


def pauli_measurement(rho: Array, N: int) -> Array:
    """
    For an n-qubit state ρ (``rho``), the function constructs all
    ``3^n`` Pauli measurements and simulates their outcomes ``N``
    times.

    Parameters
    ----------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of the measured state.
    N: int
        Number of times each measurement is repeated.
        

    Returns
    -------
    counts : (``3**n``, ``2**n``) Array
        Number of times each measurement outcome was observed. Rows
        represent measurement settings in the lexicographic order.
        For example, for ``n=2`` the order will be:

            0: XX, 1: XY, 2: XZ, 3: YX, 4: YY, 5: YZ, 6: ZX, 7: ZY, 8: ZZ.
         
        Columns represent measurement outcomes ``∈ {+, -}^n`` in the
        "lexicographic" order. For example, for ``n=2`` and every
        ``i∈[0, ..., M-1]`` one has:

            ``counts[i] = [c(+, +), c(+, -), c(-, +), c(-, -)]``,

        where c(x, x) represents number of times outcome (x, x) was observed
        for the i-th measurement setting.
    """
    xp = array_api_compat.array_namespace(rho)
    counts = Array() # dummy


    assert xp.all(xp.sum(counts, axis=-1) == N) # Sanity check
    return counts
