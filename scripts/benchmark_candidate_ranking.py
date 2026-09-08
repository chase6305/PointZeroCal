"""Benchmark contact candidate ranking with a deterministic analytical backend.

Run with PYTHONPATH=. from the repository root. The workload uses eight
contacts with 25 samples each and 100 candidates, without distance constraints.
"""

import platform
from timeit import repeat

import numpy as np

from point_zero_cal import ContactGroup, JointZeroCalibrator


class AnalyticalBackend:
    """Small nonlinear position model that isolates ranking overhead."""

    dof = 3
    joint_names = ("reference", "first", "second")

    def position_and_jacobian(self, q):
        return (
            np.array([q[0] + np.sin(q[1]), np.cos(q[1]) + q[2] ** 2, np.sin(q[2])]),
            np.array([
                [1, np.cos(q[1]), 0],
                [0, -np.sin(q[1]), 2 * q[2]],
                [0, 0, np.cos(q[2])],
            ]),
        )


def main() -> None:
    rng = np.random.default_rng(104)
    groups = [ContactGroup(str(i), rng.normal(size=(25, 3))) for i in range(8)]
    candidates = rng.normal(size=(100, 3))
    solver = JointZeroCalibrator(AnalyticalBackend())
    timings = repeat(
        lambda: solver.rank_contact_candidates(groups, candidates, group_name="3"),
        number=5, repeat=5,
    )
    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    print("8 groups x 25 samples, 100 candidates, no distance constraints")
    print(f"Median: {np.median(timings) * 1000 / 5:.2f} ms/ranking")


if __name__ == "__main__":
    main()
