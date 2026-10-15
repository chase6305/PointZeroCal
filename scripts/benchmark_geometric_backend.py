"""Benchmark UR10 geometric evaluations; run with PYTHONPATH=. from the root."""

import argparse
import platform
from pathlib import Path
from timeit import repeat

import numpy as np
import pinocchio

from point_zero_cal import GeometricParameter, PinocchioGeometricKinematics


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("count must be positive")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=positive_count, default=1000)
    parser.add_argument("--repeats", type=positive_count, default=5)
    parser.add_argument("--batch-size", type=positive_count, default=100)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="also time held-out position evaluation",
    )
    parser.add_argument(
        "--components", choices=("all", "translation", "rotation"), default="all"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    components = {
        "all": ("x", "y", "z", "roll", "pitch", "yaw"),
        "translation": ("x", "y", "z"),
        "rotation": ("roll", "pitch", "yaw"),
    }[args.components]
    backend = PinocchioGeometricKinematics(
        root / "assets/UR10/UR10.urdf",
        "ee_link",
        [
            GeometricParameter(joint, component, 0.01)
            for joint in ("Joint2", "Joint5", "jointEE")
            for component in components
        ],
        tcp_xyz=(0.08, 0, 0.16),
    )
    q = np.array([0.1, -1, 1.5, -1.2, 0.7, 0.3])
    parameters = np.linspace(-0.02, 0.02, len(backend.geometric_parameters))
    timings = repeat(
        lambda: backend.position_and_geometric_jacobian(q, parameters),
        number=args.calls,
        repeat=args.repeats,
    )
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"Pinocchio {pinocchio.__version__}"
    )
    print(f"{len(parameters)} parameters, {args.repeats} repeats of {args.calls} calls")
    print(f"Median: {np.median(timings) / args.calls * 1e6:.2f} us/call")
    configurations = np.tile(q, (args.batch_size, 1))
    configurations += np.random.default_rng(106).uniform(
        -0.1, 0.1, configurations.shape
    )

    def scalar_batch():
        positions = np.empty((args.batch_size, 3))
        jacobians = np.empty((args.batch_size, 3, len(parameters)))
        for index, configuration in enumerate(configurations):
            positions[index], jacobians[index] = (
                backend.position_and_geometric_jacobian(configuration, parameters)
            )
        return positions, jacobians

    def batch():
        return backend.position_and_geometric_jacobian_batch(configurations, parameters)

    scalar_outputs, batch_outputs = scalar_batch(), batch()
    for scalar, batched in zip(scalar_outputs, batch_outputs, strict=True):
        np.testing.assert_array_equal(scalar, batched)
    batch_calls = max(1, args.calls // args.batch_size)
    for label, operation in (
        ("Scalar loop", scalar_batch),
        ("Shared deviation", batch),
    ):
        timings = repeat(operation, number=batch_calls, repeat=args.repeats)
        per_pose = np.median(timings) / (batch_calls * args.batch_size) * 1e6
        print(f"{label}, {args.batch_size} poses/batch: {per_pose:.2f} us/pose")
    if args.position_only:

        def positions_only():
            return backend.geometric_position_batch(configurations, parameters)

        np.testing.assert_array_equal(positions_only(), batch_outputs[0])
        timings = repeat(positions_only, number=batch_calls, repeat=args.repeats)
        per_pose = np.median(timings) / (batch_calls * args.batch_size) * 1e6
        print(f"Positions only, {args.batch_size} poses/batch: {per_pose:.2f} us/pose")


if __name__ == "__main__":
    main()
