"""Benchmark complete damped SVD updates on deterministic tall systems.

Run with PYTHONPATH=. from the repository root. Timings include decomposition
and projection but exclude kinematics, residual assembly and backtracking.
"""

import platform
from timeit import repeat

import numpy as np

from point_zero_cal._least_squares import damped_svd_step


def main() -> None:
    rng = np.random.default_rng(105)
    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    for rows, columns in ((600, 5), (6000, 5), (600, 18)):
        matrix = rng.normal(size=(rows, columns))
        error = rng.normal(size=rows)
        timings = repeat(
            lambda matrix=matrix, error=error: damped_svd_step(
                matrix, error, damping=1e-10, rank_tolerance=1e-8
            ),
            number=200,
            repeat=5,
        )
        print(f"{rows} x {columns}: {np.median(timings) * 1e6 / 200:.2f} us/update")


if __name__ == "__main__":
    main()
