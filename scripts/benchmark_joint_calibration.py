"""Benchmark UR10 joint batches and complete fixed-point calibration.

Run with PYTHONPATH=. from the repository root. Dataset generation is outside
the timing. Scalar and batch paths must return identical arrays and reports.
"""

import argparse
import platform
from pathlib import Path
from timeit import repeat

import numpy as np
import pinocchio

from point_zero_cal import JointZeroCalibrator, PinocchioKinematics
from point_zero_cal.simulation import generate_fixed_point_dataset, make_rpy_samples


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("count must be positive")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_count, default=2000)
    parser.add_argument("--orientation-count", type=positive_count, default=41)
    parser.add_argument("--calls", type=positive_count, default=10)
    parser.add_argument("--repeats", type=positive_count, default=7)
    args = parser.parse_args()
    if args.orientation_count < 3 or args.orientation_count % 2 == 0:
        parser.error("--orientation-count must be odd and at least 3")
    urdf = Path(__file__).resolve().parents[1] / "assets/UR10/UR10.urdf"
    backend = PinocchioKinematics(urdf, "ee_link", tcp_xyz=(0.08, 0, 0.16))
    solver = JointZeroCalibrator(backend)
    configurations = np.random.default_rng(121).uniform(-2, 2, (args.batch_size, 6))
    dataset = generate_fixed_point_dataset(
        backend,
        np.array([0, -1.2, 1.8, -1.4, -1.57, 0]),
        make_rpy_samples(count=args.orientation_count),
    )
    truth = np.deg2rad([0, 1.2, -0.8, 1.5, -1, 2])
    measured = dataset.configurations_rad - truth
    batch = backend.position_and_jacobian_batch
    expected_arrays, expected_report = None, None
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"Pinocchio {pinocchio.__version__}"
    )
    try:
        for label in ("Scalar", "Batch"):
            backend.position_and_jacobian_batch = None if label == "Scalar" else batch
            arrays = solver._evaluate(configurations)
            fit = solver.calibrate(measured)
            assert fit.converged, fit.message
            np.testing.assert_allclose(fit.offsets_rad, truth, atol=1e-7)
            if expected_arrays is None:
                expected_arrays, expected_report = arrays, fit.to_dict()
            else:
                for original, actual in zip(expected_arrays, arrays, strict=True):
                    np.testing.assert_array_equal(actual, original)
                assert fit.to_dict() == expected_report, (
                    "batch calibration report differs"
                )
            operations = (
                (f"{args.batch_size} poses", lambda: solver._evaluate(configurations)),
                (
                    f"fit {args.orientation_count} poses",
                    lambda: solver.calibrate(measured),
                ),
            )
            for name, operation in operations:
                timings = repeat(operation, number=args.calls, repeat=args.repeats)
                print(
                    f"{label}, {name}: {np.median(timings) * 1000 / args.calls:.2f} ms"
                )
    finally:
        backend.position_and_jacobian_batch = batch


if __name__ == "__main__":
    main()
