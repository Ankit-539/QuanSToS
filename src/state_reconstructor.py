from math import log2

from array_api.latest import Array, ArrayNamespace
import array_api_compat
import numpy as np


def inverse_estimator(counts: Array) -> Array:
    """
    Estimate the state using linear inversion based on Pauli measurements
    counts.

    Parameters
    ----------
    counts : (``3**n``, ``2**n``) Array
        Counts for all Pauli measurements (see the documentation of
        `pauli_measurement` for more details).

    Returns
    -------
    rho_est : (``2**n``, ``2**n``) Array
        Estimated state.
    """
    xp = array_api_compat.array_namespace(counts)

    _, d = counts.shape
    N = xp.sum(counts[0])
    n = int(log2(d))

    p = counts / N
    c = _correlations(p, n)
    ex = _pauli_expectations(c, n)

    # Sanity checks
    assert ex[0] == 1
    if n > 1:
        assert ex[5] == sum(p[0][j] * (-1)**j.bit_count() for j in range(2**n))

    return _reconstruct_state(ex, n)


def mle(counts: Array) -> Array:
    """
    Estimate the state using maximal likelihood estimator.

    Parameters
    ----------
    counts : (``3**n``, ``2**n``) Array
        Counts for all Pauli measurements (see the documentation of
        `pauli_measurement` for more details).

    Returns
    -------
    rho_est : (``2**n``, ``2**n``) Array
        Estimated state.
    """
    pass


def _correlations(p: Array, n: int) -> Array:
    """
    Compute all correlation values for each Pauli measurement setting.

    Parameters
    ----------
    p : ``(3**n, 2**n)`` Array
        Measurement probabilities with shape. Rows correspond to
        measurement settings in lexicographic order, and
        columns correspond to measurement outcomes in lexicographic order.
    n : int
        Number of qubits.

    Returns
    -------
    correlations : (3, ..., 3, 2, ..., 2) Array
        Correlation values with shape ``(3,) * n + (2,) * n``. The first
        ``n`` axes index the measurement setting, with ``0, 1, 2``
        corresponding to ``X, Y, Z``. The last ``n`` axes indicate whether
        the measured Pauli is included in the correlation: ``0`` corresponds
        to the identity and ``1`` to the measured Pauli.

        For example, for a measurement setting ``(X, Z)``::

            correlations[0, 2, 0, 0] = <II>
            correlations[0, 2, 1, 0] = <XI>
            correlations[0, 2, 0, 1] = <IZ>
            correlations[0, 2, 1, 1] = <XZ>
    """
    xp = array_api_compat.array_namespace(p)

    p = xp.reshape(p, (3,) * n + (2,) * n)

    for q in range(n):
        axis = n + q

        plus = xp.take(p, 0, axis=axis)
        minus = xp.take(p, 1, axis=axis)

        p = xp.stack(
            (plus + minus, plus - minus),
            axis=axis,
        )

    return p


def _pauli_expectations(correlations: Array, n: int) -> Array:
    """
    Compute expectation values of all Pauli strings from measurement
    correlations.

    Parameters
    ----------
    correlations : ``(3, ..., 3, 2, ..., 2)`` Array
        Correlation values with shape ``(3,) * n + (2,) * n``. The first
        ``n`` axes index the measurement setting, with ``0, 1, 2``
        corresponding to ``X, Y, Z``. The last ``n`` axes indicate whether
        the measured Pauli is included in the correlation: ``0``
        corresponds to the identity and ``1`` to the measured Pauli.

    n : int
        Number of qubits.

    Returns
    -------
    expectations : ``(4**n,)`` Array
        Expectation values of all ``4**n`` Pauli strings with shape
        ``(4**n,)``. The Pauli strings are ordered lexicographically with
        ``I=0``, ``X=1``, ``Y=2``, and ``Z=3``. Thus, ``expectations[i]``
        contains the estimated expectation value of the Pauli string
        corresponding to the base-4 representation of ``i``.
    """
    xp = array_api_compat.array_namespace(correlations)

    T = xp.asarray([
        [[1 / 3, 0], [1 / 3, 0], [1 / 3, 0]],
        [[0, 1],    [0, 0],    [0, 0]],
        [[0, 0],    [0, 1],    [0, 0]],
        [[0, 0],    [0, 0],    [0, 1]],
    ], dtype=correlations.dtype)

    result = correlations

    for _ in range(n):
        result = xp.tensordot(
            result,
            T,
            axes=([0, result.ndim - n], [1, 2]),
        )

    result = xp.permute_dims(
        result,
        tuple(range(n - 1, -1, -1)),
    )

    return xp.reshape(result, (4**n,))


def _reconstruct_state(expectations: Array, n: int) -> Array:
    """
    Reconstruct a density matrix from Pauli expectation values.

    Parameters
    ----------
    expectations : array
        Expectation values of all Pauli strings with shape ``(4**n,)``.
        The Pauli strings are ordered lexicographically with
        ``I=0``, ``X=1``, ``Y=2``, and ``Z=3``.

    n : int
        Number of qubits.

    Returns
    -------
    rho : array
        Linear-inversion estimate of the density matrix with shape
        ``(2**n, 2**n).
    """
    xp = array_api_compat.array_namespace(expectations)

    paulis = xp.asarray([
        [[1, 0],
         [0, 1]],
        [[0, 1],
         [1, 0]],
        [[0, -1j],
         [1j, 0]],
        [[1, 0],
         [0, -1]],
    ])

    result = xp.reshape(expectations, (4,) * n)
    for _ in range(n):
        result = xp.tensordot(
            result,
            paulis,
            axes=([0], [0]),
        )

    # The tensor currently has indices
    #
    # (i1, j1, i2, j2, ..., in, jn)
    #
    # Move all row indices before all column indices.
    permutation = (
        tuple(range(0, 2*n, 2))
        + tuple(range(1, 2*n, 2))
    )

    result = xp.permute_dims(result, permutation)

    rho = xp.reshape(result, (2**n, 2**n))

    return rho / (2**n)
