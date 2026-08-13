from array_api.latest import Array
import array_api_compat
import numpy as np


def inverse_estimator(counts: Array) -> Array:
    """
    Estimate the state using linear inversion based on Pauli measurements
    counts.

    Parameters
    ----------
    counts : (``3**n``, ``2**n``) Array
        Counts for all Pauli measurements.

    Returns
    -------
    rho_est : (``2**n``, ``2**n``) Array
        Estimated state.
    """
    pass


def mle(counts: Array) -> Array:
    """
    Estimate the state using maximal likelyhood estimator.

    Parameters
    ----------
    counts : (``3**n``, ``2**n``) Array
        Counts for all Pauli measurements.

    Returns
    -------
    rho_est : (``2**n``, ``2**n``) Array
        Estimated state.
    """
    pass
