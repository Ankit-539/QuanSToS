from array_api.latest import Array
import array_api_compat
from itertools import product
import math
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

class PauliMeasurementResult:
    """
    Stores the results of a Pauli measurement.

    Parameters
    ----------
    measurements : list[str]
        List of Pauli measurement settings in lexicographic order.
    outcomes : list[str]
        List of possible measurement outcomes in lexicographic order.
    counts : Array
        Number of times each outcome was observed for each measurement.
    N : int
        Number of times each measurement was repeated.
    """

    def __init__(self, measurements, outcomes, counts, N):

        #Stores the measurement settings
        self.measurements = measurements

        #Stores the possible measurement outcomes
        self.outcomes = outcomes

        #Stores the observed measurement counts
        self.counts = counts

        #Stores the number of measurement repetitions
        self.N = N

    def __repr__(self):
        lines = []

        #Loop that combines each measurement setting with its corresponding counts
        for measurement, counts in zip(self.measurements, self.counts):
            lines.append(f"{measurement}  {counts}")

        #Combining all measurement results into a single string
        return "\n".join(lines)
    
def generate_pauli_measurements(n: int) -> list[str]:
    """
    Generates all 3^n Pauli measurement settings in lexicographic order.
    """
    return [
        ''.join(pauli_string)
        for pauli_string in product('XYZ', repeat=n)
    ]


def generate_measurement_outcomes(n: int) -> list[str]:
    """
    Generates all 2^n measurement outcomes in lexicographic order.
    """
    return [''.join(outcome_string) for outcome_string in product('+-', repeat=n)]

def to_numpy(array):
    """
    Converts an array to a NumPy array.
    """

    #NumPy array
    if isinstance(array, np.ndarray):
        return array

    #CuPy array
    if hasattr(array, "get"):
        return array.get()

    #JAX array
    if hasattr(array, "device"):
        return np.asarray(array)

    return np.asarray(array)

def multinomial(rng, N, probabilities, xp):
    """
    Sample from a multinomial distribution using the backend RNG.
    """

    # NumPy / array_api_compat NumPy
    if xp.__name__ in ("numpy", "array_api_compat.numpy"):

        probabilities_np = np.asarray(probabilities)

        counts = rng.multinomial(
            N,
            probabilities_np
        )

        return xp.asarray(
            counts,
            dtype=xp.int64
        )

    # CuPy / array_api_compat CuPy
    if xp.__name__ in ("cupy", "array_api_compat.cupy"):

        return cp.random.multinomial(
            N,
            probabilities
        )

    raise NotImplementedError(
        f"Multinomial sampling is not implemented for {xp.__name__}"
    )

def pauli_measurement(
    rho: Array,
    N: int,
    rng
) -> PauliMeasurementResult:
    """
    For an n-qubit state ρ (``rho``), construct all 3^n Pauli
    measurements and simulate their outcomes N times.

    Parameters
    ----------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of the measured state.

    N : int
        Number of times each measurement is repeated.

    rng :
        Random number generator used by ``multinomial``.

    Returns
    -------
    result : PauliMeasurementResult
        Contains the measurement settings, measurement outcomes,
        and observed counts.
    """

    # Get the array namespace associated with rho
    xp = array_api_compat.array_namespace(rho)

    # ---------------------------------------------------------
    # Find number of qubits
    # ---------------------------------------------------------

    dim = rho.shape[0]
    n = int(math.log2(dim))

    # ---------------------------------------------------------
    # Define Pauli matrices
    # ---------------------------------------------------------

    I = xp.asarray(
        [[1, 0], [0, 1]],
        dtype=rho.dtype,
        device=rho.device
    )

    X = xp.asarray(
        [[0, 1], [1, 0]],
        dtype=rho.dtype,
        device=rho.device
    )

    Y = xp.asarray(
        [[0, -1j], [1j, 0]],
        dtype=rho.dtype,
        device=rho.device
    )

    Z = xp.asarray(
        [[1, 0], [0, -1]],
        dtype=rho.dtype,
        device=rho.device
    )

    # ---------------------------------------------------------
    # Construct projectors
    #
    # projectors[pauli, outcome]
    #
    # pauli:
    #     0 -> X
    #     1 -> Y
    #     2 -> Z
    #
    # outcome:
    #     0 -> +
    #     1 -> -
    # ---------------------------------------------------------

    paulis = xp.stack([X, Y, Z])

    projectors = xp.stack(
        [
            (I + paulis) / 2,
            (I - paulis) / 2
        ],
        axis=1
    )

    # Shape:
    # (3, 2, 2, 2)
    #
    #       pauli
    #          ↓
    # projectors[p, s, i, j]
    #
    # where s is +/-

    # ---------------------------------------------------------
    # Generate measurement labels
    # ---------------------------------------------------------

    measurements = generate_pauli_measurements(n)
    outcomes = generate_measurement_outcomes(n)

    # Integer representation of Pauli strings
    #
    # e.g. XY -> (0, 1)
    #
    pauli_strings = product(range(3), repeat=n)

    count_rows = []

    # =========================================================
    # Loop over measurement SETTINGS
    # =========================================================

    for pauli_string in pauli_strings:

        # -----------------------------------------------------
        # First qubit
        # -----------------------------------------------------

        # Shape:
        #
        # (2, 2, 2)
        #
        #   outcome, row, column
        #
        projector_stack = projectors[pauli_string[0]]

        # Current number of outcomes
        num_outcomes = 2

        # Current matrix dimension
        matrix_dim = 2

        # -----------------------------------------------------
        # Add remaining qubits
        #
        # IMPORTANT:
        # There is NO loop over measurement outcomes here.
        #
        # All 2^n projectors are constructed simultaneously.
        # -----------------------------------------------------

        for pauli_idx in pauli_string[1:]:

            local_projectors = projectors[pauli_idx]

            # projector_stack:
            #
            # (num_outcomes, d, d)
            #
            # local_projectors:
            #
            # (2, 2, 2)
            #
            # Result:
            #
            # (num_outcomes, 2, d, 2, d)
            #
            combined = xp.einsum(
                "aij,bkl->abikjl",
                projector_stack,
                local_projectors
            )

            # Combine the two outcome axes
            #
            # (num_outcomes * 2, ...)
            #
            # and combine the two matrix axes:
            #
            # (d * 2, d * 2)
            #
            num_outcomes *= 2
            matrix_dim *= 2

            projector_stack = xp.reshape(
                combined,
                (
                    num_outcomes,
                    matrix_dim,
                    matrix_dim
                )
            )

        # -----------------------------------------------------
        # At this point:
        #
        # projector_stack.shape =
        #
        # (2^n, 2^n, 2^n)
        #
        # Every possible measurement outcome has its projector.
        #
        # projector_stack[k] is the projector corresponding to
        # outcome k.
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Calculate ALL probabilities simultaneously
        #
        # p_k = Tr(rho P_k)
        #
        # Instead of:
        #
        # for k in range(2**n):
        #     trace(rho @ P_k)
        #
        # we perform one tensor contraction.
        # -----------------------------------------------------

        probabilities = xp.real(
            xp.einsum(
                "ij,kji->k",
                rho,
                projector_stack
            )
        )

        # -----------------------------------------------------
        # Normalize probabilities
        # -----------------------------------------------------

        probabilities = probabilities / xp.sum(probabilities)

        # -----------------------------------------------------
        # Multinomial sampling
        # -----------------------------------------------------

        sampled_counts = multinomial(
            rng,
            N,
            probabilities,
            xp
        )

        count_rows.append(sampled_counts)

    # ---------------------------------------------------------
    # Stack results
    # ---------------------------------------------------------

    counts = xp.stack(count_rows)

    # ---------------------------------------------------------
    # Sanity check
    # ---------------------------------------------------------

    assert xp.all(
        xp.sum(counts, axis=-1) == N
    )

    # ---------------------------------------------------------
    # Return result
    # ---------------------------------------------------------

    return PauliMeasurementResult(
        measurements=measurements,
        outcomes=outcomes,
        counts=counts,
        N=N
    )

def print_counts(result: PauliMeasurementResult) -> None:
    """
    Prints the measurement counts in a readable format.
    """
    for measurement, counts in zip(result.measurements, result.counts):
        print(f"{measurement}  {counts}")

import matplotlib.pyplot as plt


def plot_counts(result: PauliMeasurementResult, measurements: str | list[str]) -> None:
    """
    Plots the measurement counts for the specified Pauli measurements.

    Parameters
    ----------
    result : PauliMeasurementResult
        Results of the Pauli measurement containing the measurement
        settings, measurement outcomes, and observed counts.
    measurements : str | list[str]
        Pauli measurement setting or list of measurement settings to plot.
    """

    #Checks if it is a valid input and converts a single measurement into a list
    if isinstance(measurements, str):
        measurements = [measurements]

    #Finding the number of measurements to plot
    num_measurements = len(measurements)

    fig, axes = plt.subplots(1, num_measurements, figsize=(5*num_measurements, 4))

    #Converting axes to an array for consistent iteration
    axes = np.atleast_1d(axes)

    #Loop that plots each requested measurement
    for measurement, ax in zip(measurements, axes):

        #Checking that the measurement exists
        if measurement not in result.measurements:
            raise ValueError(f"Measurement '{measurement}' was not found in the results.")

        #Getting the counts corresponding to the measurement
        measurement_id = result.measurements.index(measurement)
        counts = to_numpy(result.counts[measurement_id])

        #Creating the histogram
        ax.bar(result.outcomes, counts, color="steelblue")

        ax.set_title(measurement)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Counts")

        ax.set_ylim(0, max(counts) * 1.1)
        
    fig.tight_layout()
    plt.show()

