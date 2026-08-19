"""Data generation for the errors-in-variables simulations.

Every generator in this file has the same signature:

    f(n, p, sigma_a, sigma_e, beta_star, rng) -> (Z, y)

and the same model:

    X   ~ N(0, Sigma_x)     rows i.i.d., n x p
    A   ~ N(0, sigma_a^2 I) measurement error on the design
    eps ~ N(0, sigma_e^2 I) response noise
    Z = X + A,   y = X beta* + eps

Everything is Gaussian with mean zero; the generators differ only in Sigma_x,
which is what characterises the model. Each one therefore has a matching
covariance function taking p alone, and MODELS ties the two together.

Naming note: sigma_a is what the Datta & Zhang paper calls tau, and sigma_e is
what it calls sigma.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Covariance models
# ----------------------------------------------------------------------

def sigma_x_ar(p: int) -> np.ndarray:
    """Autoregressive: Sigma_x[i, j] = 0.5^|i-j|."""
    idx = np.arange(p)
    return 0.5 ** np.abs(idx[:, None] - idx[None, :])


def sigma_x_cs(p: int) -> np.ndarray:
    """Compound symmetry: Sigma_x[i, j] = 0.5 + 0.5 * I(i == j).

    Nearly rank one for large p -- its top eigenvalue is about p/2 -- so it is
    far harder for support recovery than AR at the same nominal correlation.
    """
    return 0.5 * np.ones((p, p)) + 0.5 * np.eye(p)


def sigma_x_identity(p: int) -> np.ndarray:
    """Uncorrelated design."""
    return np.eye(p)


# ----------------------------------------------------------------------
# Generators -- one per model, all sharing the same API
# ----------------------------------------------------------------------

def _generate(sigma_x: np.ndarray, n: int, p: int, sigma_a: float,
              sigma_e: float, beta_star: np.ndarray,
              rng: np.random.Generator) -> tuple:
    beta_star = np.asarray(beta_star, dtype=float)
    if beta_star.shape != (p,):
        raise ValueError(f"beta_star has shape {beta_star.shape}, expected ({p},)")
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky(sigma_x).T
    y = X @ beta_star + rng.normal(scale=sigma_e, size=n)
    Z = X + rng.normal(scale=sigma_a, size=(n, p))
    return Z, y


def ar(n, p, sigma_a, sigma_e, beta_star, rng):
    """Sigma_x[i, j] = 0.5^|i-j|."""
    return _generate(sigma_x_ar(p), n, p, sigma_a, sigma_e, beta_star, rng)


def cs(n, p, sigma_a, sigma_e, beta_star, rng):
    """Sigma_x = 0.5 + 0.5 I."""
    return _generate(sigma_x_cs(p), n, p, sigma_a, sigma_e, beta_star, rng)


def identity(n, p, sigma_a, sigma_e, beta_star, rng):
    """Sigma_x = I."""
    return _generate(sigma_x_identity(p), n, p, sigma_a, sigma_e, beta_star, rng)


MODELS = {
    "ar": (ar, sigma_x_ar),
    "cs": (cs, sigma_x_cs),
    "identity": (identity, sigma_x_identity),
}


def get_model(name: str):
    """(generator, sigma_x) for a model name."""
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; available: {sorted(MODELS)}") from None


# ----------------------------------------------------------------------
# beta* presets
# ----------------------------------------------------------------------

def _beta_datta_zhang(p: int) -> np.ndarray:
    """(3, 1.5, 0, 0, 2, 0, ..., 0)', the additive-errors case of the paper."""
    b = np.zeros(p)
    b[0], b[1], b[4] = 3.0, 1.5, 2.0
    return b


def _beta_two_tier(p: int, n_strong: int = 3, n_weak: int = 3,
                   strong: float = 3.0, weak: float = 0.5,
                   spacing: int | None = None) -> np.ndarray:
    """Strong and weak coefficients, spread out so they are near-uncorrelated.

    The case the trimmed lasso exists for: a uniform lasso penalty erases the
    weak-but-real coefficients along with the nulls.
    """
    k = n_strong + n_weak
    if spacing is None:
        spacing = max(p // k, 1)
    pos = (np.arange(k) * spacing) % p
    if len(set(pos.tolist())) != k:
        raise ValueError(f"spacing={spacing} collides for p={p}, k={k}")
    b = np.zeros(p)
    b[pos[:n_strong]] = strong
    b[pos[n_strong:]] = weak
    return b


def _beta_equal(p: int, k: int = 5, value: float = 2.0,
                spacing: int | None = None) -> np.ndarray:
    """k equal coefficients, spread out."""
    return _beta_two_tier(p, n_strong=k, n_weak=0, strong=value,
                          spacing=spacing)


BETA_PRESETS = {
    "datta_zhang": _beta_datta_zhang,
    "two_tier": _beta_two_tier,
    "equal": _beta_equal,
}


def make_beta(preset: str, p: int, **kw) -> np.ndarray:
    """Build beta* from a named preset."""
    try:
        fn = BETA_PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"unknown beta preset {preset!r}; available: {sorted(BETA_PRESETS)}"
        ) from None
    return fn(p, **kw)


# ----------------------------------------------------------------------
# Design diagnostics
# ----------------------------------------------------------------------

def snr(model: str, p: int, beta_star: np.ndarray, sigma_e: float) -> float:
    """Var(x' beta*) / sigma_e^2 = beta*' Sigma_x beta* / sigma_e^2.

    Depends on Sigma_x, beta* and sigma_e at once, so it is the single best
    check that a design is coded correctly. The Datta & Zhang additive-errors
    case gives 2.36 for AR and 3.20 for CS.
    """
    _, sigma_x = get_model(model)
    b = np.asarray(beta_star, dtype=float)
    return float(b @ sigma_x(p) @ b) / sigma_e ** 2


def signal_power(model: str, p: int, beta_star: np.ndarray) -> float:
    """beta*' Sigma_x beta*."""
    _, sigma_x = get_model(model)
    b = np.asarray(beta_star, dtype=float)
    return float(b @ sigma_x(p) @ b)


if __name__ == "__main__":
    p = 250
    b = make_beta("datta_zhang", p)
    print(f"beta* nonzeros: {dict(zip(*np.nonzero(b), b[np.nonzero(b)]))}")
    for m in ("ar", "cs"):
        print(f"  {m:>8}: signal = {signal_power(m, p, b):7.3f}   "
              f"SNR = {snr(m, p, b, 3.0):.4f}   (paper: AR 2.36, CS 3.20)")
