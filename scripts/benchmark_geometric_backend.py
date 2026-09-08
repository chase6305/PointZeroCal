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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    backend = PinocchioGeometricKinematics(
        root / "assets/UR10/UR10.urdf",
        "ee_link",
        [
            GeometricParameter(joint, component, 0.01)
            for joint in ("Joint2", "Joint5", "jointEE")
            for component in ("x", "y", "z", "roll", "pitch", "yaw")
        ],
        tcp_xyz=(0.08, 0, 0.16),
    )
    q = np.array([0.1, -1, 1.5, -1.2, 0.7, 0.3])
    parameters = np.linspace(-0.02, 0.02, 18)
    timings = repeat(
        lambda: backend.position_and_geometric_jacobian(q, parameters),
        number=args.calls,
        repeat=args.repeats,
    )
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"Pinocchio {pinocchio.__version__}"
    )
    print(f"18 parameters, {args.repeats} repeats of {args.calls} calls")
    print(f"Median: {np.median(timings) / args.calls * 1e6:.2f} us/call")


if __name__ == "__main__":
    main()
