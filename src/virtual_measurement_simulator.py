from array_api.latest import Array
import array_api_compat
from itertools import product
import math
import numpy as np
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
    Draw samples from a multinomial distribution.

    Parameters
    ----------
    rng : backend-specific random number generator
        Random number generator for the selected backend.
    N : int
        Number of samples.
    probabilities : Array
        Probability vector.
    xp : ArrayNamespace
        Array API namespace.

    Returns
    -------
    counts : Array
        Sampled counts.
    """

    if xp.__name__ == "numpy":
        return xp.asarray(rng.multinomial(N, probabilities))

    if xp.__name__ == "cupy":
        return rng.multinomial(N, probabilities)

    raise NotImplementedError(f"Multinomial sampling is not implemented for {xp.__name__}")

def pauli_measurement(rho: Array, N: int) -> PauliMeasurementResult:
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
    result : PauliMeasurementResult
        Contains the measurement settings, measurement outcomes, and
        observed counts.

        ``result.measurements`` contains the measurement settings in
        lexicographic order. For example, for ``n=2``:

            0: XX, 1: XY, 2: XZ, 3: YX, 4: YY, 5: YZ, 6: ZX, 7: ZY, 8: ZZ.

        ``result.outcomes`` contains the measurement outcomes in
        lexicographic order. For example, for ``n=2``:

            0: ++, 1: +-, 2: -+, 3: --.

        ``result.counts`` is a (``3**n``, ``2**n``) Array where
        ``result.counts[i, j]`` is the number of times outcome ``j``
        was observed for measurement setting ``i``.
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
    projectors = xp.stack([(I + pauli_stacked) / 2,
                           (I - pauli_stacked) / 2], axis=1)

    #Defining Count Size
    count_size = (3**n, 2**n)

    #Defining the Pauli strings
    pauli_strings = list(product(range(3), repeat=n))
    output_strings = list(product(range(2), repeat=n))

    #Defining the nicely readable measurement and outcome strings
    measurements = generate_pauli_measurements(n)
    outcomes = generate_measurement_outcomes(n)

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
        numpy_probabilities = to_numpy(probabilities)

        sampled_counts_np = np.random.multinomial(N, numpy_probabilities)

        sampled_counts_xp = xp.asarray(sampled_counts_np, dtype=xp.int64, device=rho.device)

        count_rows.append(sampled_counts_xp)

    counts = xp.stack(count_rows)

    # Sanity check
    assert xp.all(xp.sum(counts, axis=-1) == N)

    return PauliMeasurementResult(measurements=measurements, outcomes=outcomes,counts=counts, N=N)

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

