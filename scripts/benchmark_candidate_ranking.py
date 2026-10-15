"""Benchmark contact candidate ranking with a deterministic analytical backend.

Run with PYTHONPATH=. from the repository root. The workload uses eight
contacts with 25 samples each and 100 candidates by default. Optional chain
or complete distance constraints exercise the grouped ranking cache.
"""

import argparse
import platform
from timeit import repeat

import numpy as np

from point_zero_cal import ContactGroup, JointZeroCalibrator, KnownDistance


class AnalyticalBackend:
    """Small nonlinear position model that isolates ranking overhead."""

    dof = 3
    joint_names = ("reference", "first", "second")

    def position_and_jacobian(self, q):
        return (
            np.array([q[0] + np.sin(q[1]), np.cos(q[1]) + q[2] ** 2, np.sin(q[2])]),
            np.array(
                [
                    [1, np.cos(q[1]), 0],
                    [0, -np.sin(q[1]), 2 * q[2]],
                    [0, 0, np.cos(q[2])],
                ]
            ),
        )


class MultiJointBackend:
    """Configurable analytic model for larger candidate information systems."""

    def __init__(self, dof: int):
        self.dof = dof
        self.joint_names = tuple(f"joint_{index}" for index in range(dof))
        self.basis = np.random.default_rng(105).normal(size=(3, dof))

    def position_and_jacobian(self, q):
        return self.basis @ np.sin(q), self.basis * np.cos(q)


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("count must be positive")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=positive_count, default=8)
    parser.add_argument("--samples", type=positive_count, default=25)
    parser.add_argument("--candidates", type=positive_count, default=100)
    parser.add_argument("--dof", type=positive_count, default=3)
    parser.add_argument("--calls", type=positive_count, default=5)
    parser.add_argument("--repeats", type=positive_count, default=5)
    parser.add_argument(
        "--distances", choices=("none", "chain", "complete"), default="none"
    )
    args = parser.parse_args()
    if args.samples < 2 or args.dof < 2:
        parser.error("--samples and --dof must each be at least 2")
    if args.distances != "none" and args.groups < 2:
        parser.error("distance constraints require at least two groups")
    rng = np.random.default_rng(104)
    groups = [
        ContactGroup(str(i), rng.normal(size=(args.samples, args.dof)))
        for i in range(args.groups)
    ]
    candidates = rng.normal(size=(args.candidates, args.dof))
    backend = AnalyticalBackend() if args.dof == 3 else MultiJointBackend(args.dof)
    solver = JointZeroCalibrator(backend)
    target = str(min(3, args.groups - 1))
    pairs = (
        [(index, index + 1) for index in range(args.groups - 1)]
        if args.distances == "chain"
        else [
            (first, second)
            for first in range(args.groups)
            for second in range(first + 1, args.groups)
        ]
        if args.distances == "complete"
        else []
    )
    distances = [
        KnownDistance(str(first), str(second), 0.05) for first, second in pairs
    ]
    timings = repeat(
        lambda: solver.rank_contact_candidates(
            groups, candidates, group_name=target, distances=distances
        ),
        number=args.calls,
        repeat=args.repeats,
    )
    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    print(
        f"{args.groups} groups x {args.samples} samples, {args.candidates} candidates, "
        f"{args.dof - 1} active joints, {len(distances)} distance constraints"
    )
    print(f"Median: {np.median(timings) * 1000 / args.calls:.2f} ms/ranking")


if __name__ == "__main__":
    main()
