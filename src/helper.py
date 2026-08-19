from array_api.latest import Array, ArrayNamespace
import array_api_compat
import numpy as np
import cupy as cp
import time

def bloch_vector(rho):
    """
    Convert a single-qubit density matrix to its Bloch vector.
    """
    xp = array_api_compat.array_namespace(rho)
    X = xp.asarray([[0, 1], [1, 0]], dtype=rho.dtype)
    Y = xp.asarray([[0, -1j], [1j, 0]], dtype=rho.dtype)
    Z = xp.asarray([[1, 0], [0, -1]], dtype=rho.dtype)

    x = xp.real(xp.trace(rho @ X))
    y = xp.real(xp.trace(rho @ Y))
    z = xp.real(xp.trace(rho @ Z))

    return xp.stack([x, y, z])

def benchmark_function(func, n_values, xp, rng, repeats=10):
    """
    Benchmark a state-generation function for different numbers of qubits.

    Parameters
    ----------
    func : callable
        State generation function.
    n_values : iterable
        Numbers of qubits to benchmark.
    xp : ArrayNamespace
        NumPy or CuPy namespace.
    rng :
        Random number generator.
    repeats : int
        Number of repetitions for each n.

    Returns
    -------
    times : list[float]
        Average runtime in milliseconds.
    """

    times = []

    for n in n_values:

        # Warm-up
        rho = func(n, xp, rng)

        if xp is cp:
            cp.cuda.Stream.null.synchronize()

        run_times = []

        for _ in range(repeats):

            if xp is cp:
                start = cp.cuda.Event()
                end = cp.cuda.Event()

                start.record()

                rho = func(n, xp, rng)

                end.record()
                end.synchronize()

                elapsed_ms = cp.cuda.get_elapsed_time(start, end)

            else:
                start = time.perf_counter()

                rho = func(n, xp, rng)

                end = time.perf_counter()

                elapsed_ms = (end - start) * 1000

            run_times.append(elapsed_ms)

        times.append(np.mean(run_times))

    return times

from time import perf_counter


def benchmark_pauli_measurement(func, rho, N, xp, repeats=5):
    """
    Benchmark a Pauli measurement function.

    Parameters
    ----------
    func : callable
        Pauli measurement function.
    rho : Array
        Density matrix to measure.
    N : int
        Number of measurement repetitions.
    xp : ArrayNamespace
        Array API namespace.
    repeats : int
        Number of benchmark repetitions.

    Returns
    -------
    float
        Mean runtime in milliseconds.
    """

    func(rho, N)

    times = []

    for _ in range(repeats):

        if xp.__name__ == "cupy":
            xp.cuda.Stream.null.synchronize()

        start = perf_counter()

        func(rho, N)

        if xp.__name__ == "cupy":
            xp.cuda.Stream.null.synchronize()

        end = perf_counter()

        times.append((end - start) * 1000)

    return np.mean(times)