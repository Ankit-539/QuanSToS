from array_api.latest import Array
import array_api_compat
from itertools import product
import math
import numpy as np


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

        where c(x, y) represents number of times outcome (x, y) was observed
        with the i-th measurement setting.
    """
    xp = array_api_compat.array_namespace(rho)

    #finding number of qubits from density matrix
    dim = rho.shape[0]
    n = int(math.log2(dim))

    #Defining the Pauli Matrices
    I = xp.asarray([[1, 0], [0, 1]], dtype=rho.dtype, device=rho.device)
    X = xp.asarray([[0, 1], [1, 0]], dtype=rho.dtype, device=rho.device)
    Y = xp.asarray([[0, -1j], [1j, 0]], dtype=rho.dtype, device=rho.device)
    Z = xp.asarray([[1, 0], [0, -1]], dtype=rho.dtype, device=rho.device)

    #getting an array of corresponding projectors in order (X+, X-, Y+, Y-, Z+, Z-)
    pauli_stacked = xp.stack([X, Y, Z])
    projectors = xp.stack([(I+pauli_stacked)/2, (I-pauli_stacked)/2], axis=1)

    #Defining Count Size
    count_size = (3**n, 2**n)

    #Defining the Pauli strings
    pauli_strings = list(product(range(3), repeat=n))
    output_strings = list(product(range(2), repeat=n))

    count_rows = []
    #Loop that iterates over each string to calculate probabilities observed
    for setting_id, pauli_string in enumerate(pauli_strings):

        probabilities = []
        #Loop that goes overall all possible outcomes for a given input
        for outcome_id, output_string in enumerate(output_strings):

            #stores the projector corresponding to the pauli string
            projector = projectors[pauli_string[0], output_string[0]]
            for pauli_idx, outcome_idx in zip(pauli_string[1:], output_string[1:]):
                projector = xp.kron(projector, projectors[pauli_idx, outcome_idx])

            prob = xp.trace(xp.matmul(rho, projector))
            probabilities.append(xp.real(prob))

        probabilities = xp.stack(probabilities)

        # Normalize probabilities
        probabilities = probabilities / xp.sum(probabilities)

        # Multinomial sampling
        numpy_probabilities = np.asarray(probabilities)
        sampled_counts_np = np.random.multinomial(N, numpy_probabilities.tolist())
        sampled_counts_xp = xp.asarray(sampled_counts_np, dtype=xp.int64, device=rho.device)

        count_rows.append(sampled_counts_xp)

    counts = xp.stack(count_rows)

    assert xp.all(xp.sum(counts, axis=-1) == N) # Sanity check
    return counts