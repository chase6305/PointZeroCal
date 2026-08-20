"""Command-line interface for PointZeroCal."""

import argparse
import json
from pathlib import Path

import numpy as np

from .calibrator import JointZeroCalibrator
from .pinocchio_backend import PinocchioKinematics


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(description="Calibrate robot joint zero offsets")
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--end-frame", required=True)
    parser.add_argument("--samples", required=True, type=Path, help="CSV with one pose per row")
    parser.add_argument("--unit", choices=("deg", "rad"), default="deg")
    parser.add_argument(
        "--tcp-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 0.0)
    )
    parser.add_argument(
        "--tcp-rpy", nargs=3, type=float, metavar=("R", "P", "Y"), default=(0.0, 0.0, 0.0),
        help="TCP rotation in radians relative to end-frame",
    )
    parser.add_argument(
        "--fixed-joint",
        action="append",
        help="joint name to hold fixed (defaults to the first actuated joint)",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run calibration and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        samples = np.loadtxt(args.samples, delimiter=",", ndmin=2)
        backend = PinocchioKinematics(
            args.urdf,
            args.end_frame,
            tcp_xyz=tuple(args.tcp_xyz),
            tcp_rpy=tuple(args.tcp_rpy),
        )
        fixed_joints = args.fixed_joint if args.fixed_joint is not None else (0,)
        result = JointZeroCalibrator(
            backend, max_iterations=args.max_iterations
        ).calibrate(samples, sample_unit=args.unit, fixed_joints=fixed_joints)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    # JSON has no representation for infinity. Rank-deficient condition numbers
    # are therefore emitted as null instead of the non-standard Infinity token.
    condition_number = (
        result.condition_number if np.isfinite(result.condition_number) else None
    )
    payload = {
        "converged": result.converged,
        "message": result.message,
        "sample_count": result.sample_count,
        "joint_names": backend.joint_names,
        "active_joint_names": result.active_joint_names,
        "offsets_rad": result.offsets_rad.tolist(),
        "offsets_deg": result.offsets_deg.tolist(),
        "iterations": result.iterations,
        "residual_rms": result.residual_rms,
        "initial_point_rms": result.initial_point_rms,
        "corrected_point_rms": result.corrected_point_rms,
        "step_norm": result.step_norm,
        "rank": result.rank,
        "condition_number": condition_number,
        "singular_values": result.singular_values.tolist(),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        try:
            args.output.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            parser.error(f"cannot write output: {exc}")
    return 0 if result.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
