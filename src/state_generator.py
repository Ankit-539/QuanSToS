from __future__ import annotations

from array_api.latest import Array, ArrayNamespace
import numpy as np


def get_random_product(n: int, xp: ArrayNamespace) -> Array:
    """
    Generate a random product state on ``n`` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.
    xp : ArrayNamespace
            Array backend namespace.

    Returns
    -------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of a random state.
    """
    psi_product = xp.asarray([1.0 + 0j])
    rng = RNG(xp)
    for _ in range(n):
        u = rng.uniform(1, -1.0, 1.0)[0]
        phi = rng.uniform(1, 0.0, 2.0 * xp.pi)[0]
        u = xp.asarray(u)
        phi = xp.asarray(phi)
        theta = xp.arccos(u)
        psi = xp.stack([
            xp.cos(theta / 2), xp.exp(1j * phi) * xp.sin(theta / 2)
        ])
        psi_product = xp.reshape(psi_product[:, None] * psi[None, :], (-1,))
    rho = xp.outer(psi_product, xp.conj(psi_product))
    return rho


def get_random_haar(n: int, xp: ArrayNamespace) -> Array:
    """
    Generate a Haar-random state on ``n`` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.
    xp : ArrayNamespace
        Array backend namespace.

    Returns
    -------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of a random state.
    """
    rng = RNG(xp)
    dimension = 2**n
    real = rng.normal(dimension, 0.0, 1.0)
    imaginary = rng.normal(dimension, 0.0, 1.0)
    real = xp.asarray(real)
    imaginary = xp.asarray(imaginary)
    z = real + 1j * imaginary
    psi_haar = psi_haar = z / xp.sqrt(xp.sum(xp.abs(z) ** 2))
    rho = xp.outer(psi_haar, xp.conj(psi_haar))
    return rho


def get_random_mixed(n: int, xp: ArrayNamespace) -> Array:
    """
    Generate a random mixed state on ``n`` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of a random state.
    """
    dimension = 2**n
    rng = RNG(xp)
    real = rng.normal((dimension, dimension), 0.0, 1.0)
    imaginary = rng.normal((dimension, dimension), 0.0, 1.0)
    real = xp.asarray(real)
    imaginary = xp.asarray(imaginary)
    z = real + 1j * imaginary
    rho = xp.matmul(z, xp.conj(xp.matrix_transpose(z)))
    rho = rho / xp.trace(rho)
    return rho


def get_random_product_pole_biased(n: int, xp: ArrayNamespace) -> Array:
    """
    Generate a random product state on ``n`` qubits, with each
    single-qubit state sampled uniformly in theta and phi.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    rho : (``2**n``, ``2**n``) Array
        Density matrix of a random state.
    """
    psi_product = xp.asarray([1.0 + 0j])
    rng = RNG(xp)

    for _ in range(n):
        theta = rng.uniform(1, 0.0, np.pi)[0]
        phi = rng.uniform(1, 0.0, 2.0 * np.pi)[0]

        theta = xp.asarray(theta)
        phi = xp.asarray(phi)

        psi = xp.stack([xp.cos(theta/2), xp.exp(1j*phi)*xp.sin(theta/2)])
        psi_product = xp.kron(psi_product, psi)

    rho = xp.outer(psi_product, xp.conj(psi_product))

    return rho


class RNG:
    """Backend-independent random number generator.

    Supported backends:
        - NumPy
        - CuPy
        - JAX

    The returned arrays belong to the corresponding backend.
    """

    def __init__(self, xp: ArrayNamespace, seed: int | None = None):
        self.xp = xp
        name = xp.__name__.split(".")[0]

        if name == "numpy":
            self.backend = "numpy"
            self._rng = xp.random.default_rng(seed)

        elif name in xp.__name__.split("."):
            self.backend = "cupy"
            self._rng = xp.random.default_rng(seed)

        elif name == "jax":
            import jax

            self.backend = "jax"
            self._key = jax.random.key(0 if seed is None else seed)

        else:
            raise TypeError(f"Unsupported array namespace: {xp.__name__}")

    def _jax_key(self):
        import jax

        self._key, key = jax.random.split(self._key)
        return key

    def normal(
        self,
        size,
        loc: float = 0.0,
        scale: float = 1.0,
        dtype=None,
    ):
        """Draw samples from N(loc, scale²)."""

        if self.backend == "numpy":
            return self._rng.normal(
                loc=loc,
                scale=scale,
                size=size,
            ).astype(dtype) if dtype is not None else self._rng.normal(
                loc=loc,
                scale=scale,
                size=size,
            )

        if self.backend == "cupy":
            x = self._rng.standard_normal(size)
            x = x * scale + loc
            return x.astype(dtype) if dtype is not None else x

        # JAX
        import jax

        x = jax.random.normal(
            self._jax_key(),
            shape=size,
            dtype=dtype,
        )
        return x * scale + loc

    def standard_normal(self, size, dtype=None):
        """Draw samples from N(0, 1)."""
        return self.normal(size, dtype=dtype)

    def uniform(
        self,
        size,
        low: float = 0.0,
        high: float = 1.0,
        dtype=None,
    ):
        """Draw samples uniformly from [low, high)."""

        if self.backend == "numpy":
            return self._rng.uniform(
                low=low,
                high=high,
                size=size,
            ).astype(dtype) if dtype is not None else self._rng.uniform(
                low=low,
                high=high,
                size=size,
            )

        if self.backend == "cupy":
            x = self._rng.uniform(low, high, size)
            return x.astype(dtype) if dtype is not None else x

        # JAX
        import jax

        return jax.random.uniform(
            self._jax_key(),
            shape=size,
            minval=low,
            maxval=high,
            dtype=dtype,
        )

    def integers(
        self,
        size,
        low: int,
        high: int | None = None,
        dtype=None,
    ):
        """Draw random integers from [low, high)."""

        if self.backend == "numpy":
            return self._rng.integers(
                low=low,
                high=high,
                size=size,
                dtype=dtype,
            )

        if self.backend == "cupy":
            x = self._rng.integers(
                low=low,
                high=high,
                size=size,
            )
            return x.astype(dtype) if dtype is not None else x

        # JAX
        import jax

        if high is None:
            low, high = 0, low

        return jax.random.randint(
            self._jax_key(),
            shape=size,
            minval=low,
            maxval=high,
            dtype=dtype,
        )

    def random(self, size, dtype=None):
        """Draw samples uniformly from [0, 1)."""
        return self.uniform(size, dtype=dtype)

    def complex_normal(self, size, dtype=None):
        """Draw standard complex Gaussian samples.

        Each component has E[|z|²] = 1.
        """
        if dtype is None:
            dtype = self.xp.complex128

        real_dtype = self.xp.float64
        return (
            self.standard_normal(size, dtype=real_dtype)
            + 1j * self.standard_normal(size, dtype=real_dtype)
        ) / self.xp.sqrt(2)

    def choice(self, a, size=None, replace=True):
        """Random choice from an array or range(a)."""

        if self.backend == "numpy":
            return self._rng.choice(a, size=size, replace=replace)

        if self.backend == "cupy":
            return self._rng.choice(a, size=size, replace=replace)

        # JAX
        import jax

        if not hasattr(a, "shape"):
            a = self.xp.arange(a)

        return jax.random.choice(
            self._jax_key(),
            a,
            shape=size,
            replace=replace,
        )
