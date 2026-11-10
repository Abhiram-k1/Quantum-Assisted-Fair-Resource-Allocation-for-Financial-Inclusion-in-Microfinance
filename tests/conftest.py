import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qfar.config import load_config  # noqa: E402
from qfar.qubo import AllocationQUBO  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def psd_matrix(rng, N, scale=0.3):
    A = rng.uniform(-scale, scale, (N, N))
    return A @ A.T / N


def rand_qubo(seed, N=7, eta=0.0, lam_F=0.0, budget_frac=0.5, w_range=(1, 11), lam_B=None):
    r = np.random.default_rng(seed)
    U = r.uniform(.05, 1.0, N)
    w = r.integers(*w_range, N)
    g = np.array([0, 1] * (N // 2) + [0] * (N % 2))
    r.shuffle(g)
    B = max(1, int(w.sum() * budget_frac))
    return AllocationQUBO(U, w, B, g, psd_matrix(r, N), eta, lam_F, lam_B=lam_B)
