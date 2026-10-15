"""Benchmark complete held-out reports and peak traced allocations.

Run with PYTHONPATH=. from the repository root. Random UR10 configurations
measure FK and report generation, rather than physical contact accuracy.
Compare optional position-only methods with the existing Jacobian fallback.
"""

import argparse
import platform
import tracemalloc
from functools import partial
from pathlib import Path
from timeit import repeat

import numpy as np
import pinocchio

from point_zero_cal import (
    ContactGroup,
    GeometricCalibrator,
    GeometricParameter,
    JointZeroCalibrator,
    KnownDistance,
    PinocchioGeometricKinematics,
    PinocchioKinematics,
)


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("count must be positive")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive_count, default=5000)
    parser.add_argument("--groups", type=positive_count, default=4)
    parser.add_argument("--calls", type=positive_count, default=3)
    parser.add_argument("--repeats", type=positive_count, default=5)
    parser.add_argument(
        "--mode", choices=("joint", "geometric", "both"), default="both"
    )
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    urdf = Path(__file__).resolve().parents[1] / "assets/UR10/UR10.urdf"
    rng = np.random.default_rng(118)
    groups = [
        ContactGroup(str(index), rng.uniform(-2, 2, (args.samples, 6)))
        for index in range(args.groups)
    ]
    distances = [
        KnownDistance(str(first), str(second), 1)
        for first in range(args.groups)
        for second in range(first + 1, args.groups)
    ]
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"Pinocchio {pinocchio.__version__}"
    )
    print(
        f"{args.groups} groups x {args.samples} poses, "
        f"{len(distances)} distance constraints"
    )
    for mode in ("joint", "geometric"):
        if args.mode not in (mode, "both"):
            continue
        if mode == "geometric":
            parameters = [
                GeometricParameter(joint, component, 0.01)
                for joint in ("Joint2", "Joint5", "jointEE")
                for component in ("x", "y", "z", "roll", "pitch", "yaw")
            ]
            backend = PinocchioGeometricKinematics(
                urdf, "ee_link", parameters, tcp_xyz=(0.08, 0, 0.16)
            )
            solver = GeometricCalibrator(backend)
            values = np.linspace(-0.02, 0.02, len(parameters))
            method = "geometric_position_batch"

            evaluate = partial(
                solver.evaluate_groups, groups, values, distances=distances
            )
        else:
            backend = PinocchioKinematics(urdf, "ee_link", tcp_xyz=(0.08, 0, 0.16))
            solver = JointZeroCalibrator(backend)
            method = "position_batch"

            evaluate = partial(
                solver.evaluate_groups,
                groups,
                offsets_rad=np.linspace(-0.02, 0.02, 6),
                distances=distances,
            )

        batch = getattr(backend, method)
        expected = None
        try:
            for label in ("Full Jacobians", "Positions only"):
                setattr(backend, method, None if label == "Full Jacobians" else batch)
                report = evaluate().to_dict()
                if expected is None:
                    expected = report
                else:
                    assert report == expected, (
                        "position-only report differs from fallback"
                    )
                timings = repeat(evaluate, number=args.calls, repeat=args.repeats)
                tracemalloc.start()
                try:
                    evaluate()
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                print(
                    f"{mode}, {label}: {np.median(timings) * 1000 / args.calls:.2f} "
                    f"ms/report; peak traced allocation {peak / 1e6:.2f} MB"
                )
        finally:
            setattr(backend, method, batch)


if __name__ == "__main__":
    main()
