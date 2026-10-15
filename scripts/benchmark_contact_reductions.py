"""Benchmark residual assembly, excluding kinematics evaluation.

Run with PYTHONPATH=. from the repository root. Synthetic cached positions
and Jacobians isolate centring, weighting and residual reduction overhead;
these timings do not measure complete calibration speed.
"""

import argparse
import platform
import tracemalloc
from timeit import repeat

import numpy as np

from point_zero_cal import ContactGroup, JointZeroCalibrator, _groups


class CachedBackend:
    def __init__(self, dof):
        self.dof = dof
        self.joint_names = tuple(f"joint_{index}" for index in range(dof))


class CachedCalibrator(JointZeroCalibrator):
    """Supply deterministic cached outputs instead of forward kinematics."""

    def __init__(self, positions, jacobians):
        super().__init__(CachedBackend(jacobians.shape[-1]))
        self.positions = positions
        self.jacobians = jacobians

    def _evaluate(self, configurations):
        return self.positions, self.jacobians


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("count must be positive")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=positive_count, default=8)
    parser.add_argument("--samples", type=positive_count, default=25)
    parser.add_argument("--dof", type=positive_count, default=6)
    parser.add_argument("--calls", type=positive_count, default=1000)
    parser.add_argument("--repeats", type=positive_count, default=5)
    parser.add_argument(
        "--memory", action="store_true", help="also trace peak allocations"
    )
    args = parser.parse_args()
    if args.samples < 2 or args.dof < 2:
        parser.error("--samples and --dof must each be at least 2")
    total = args.groups * args.samples
    rng = np.random.default_rng(104)
    configurations = rng.normal(size=(total, args.dof))
    positions = rng.normal(size=(total, 3))
    jacobians = rng.normal(size=(total, 3, args.dof))
    active = np.arange(1, args.dof)
    solver = CachedCalibrator(positions, jacobians)
    _, plan = _groups.prepare_groups(
        [
            ContactGroup(str(i), configurations[i * args.samples : (i + 1) * args.samples])
            for i in range(args.groups)
        ],
        (),
        "rad",
        args.dof,
    )
    grouped_label = f"{args.groups} contacts x {args.samples} samples, no distances"
    workloads = {
        f"Single contact, {total} samples": lambda: solver._single_state(
            configurations, active
        ),
        grouped_label: lambda: _groups.group_state_system(
            positions, jacobians, plan, active
        ),
    }
    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    print("Residual assembly only; kinematics and SVD excluded")
    print(f"{args.dof - 1} active joints; {args.calls} calls x {args.repeats} repeats")
    for label, operation in workloads.items():
        timings = repeat(operation, number=args.calls, repeat=args.repeats)
        print(f"{label}: {np.median(timings) * 1e6 / args.calls:.2f} us/evaluation")
        if args.memory:
            tracemalloc.start()
            operation()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(f"  Peak traced allocations: {peak} bytes")


if __name__ == "__main__":
    main()
