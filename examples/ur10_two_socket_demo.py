"""Fit two simulated sockets and validate fixed offsets at another placement."""

import argparse
import io
import json
from pathlib import Path

import numpy as np

from point_zero_cal import (
    ContactGroup,
    JointZeroCalibrator,
    KnownDistance,
    PinocchioKinematics,
)
from point_zero_cal._io import write_text_atomic
from point_zero_cal._validation import finite_scalar
from point_zero_cal.simulation import generate_fixed_point_dataset, make_rpy_samples

ROOT = Path(__file__).resolve().parents[1]


def run_demo(*, count: int = 21, encoder_noise_deg: float = 0.002):
    """Return synthetic observations and fit/validation reports; no hardware I/O."""
    noise = finite_scalar(encoder_noise_deg, "encoder_noise_deg", allow_zero=True)
    backend = PinocchioKinematics(
        ROOT / "assets/UR10/UR10.urdf", "ee_link", tcp_xyz=(0.08, 0, 0.16)
    )
    seed = np.array([0, -1.2, 1.8, -1.4, -1.57, 0])
    origin = np.array(backend.tcp_pose(seed).translation)
    expected = np.deg2rad([0, 1.2, -0.8, 1.5, -1, 2])
    deltas = make_rpy_samples(count=count, max_angle_deg=40)
    rng = np.random.default_rng(42)
    groups = []
    placements = {
        "front/0": [0, 0, 0],
        "front/1": [0.05, 0, 0],
        "left/0": [0, 0.08, 0.03],
        "left/1": [0.05, 0.08, 0.03],
    }
    for name, shift in placements.items():
        dataset = generate_fixed_point_dataset(
            backend, seed, deltas, target_xyz=origin + shift
        )
        measured = dataset.configurations_rad - expected
        measured += rng.normal(0, np.deg2rad(noise), measured.shape)
        groups.append(ContactGroup(name, measured))
    # The same physical fixture has a 50 mm spacing at both placements.
    training_distance = KnownDistance("front/0", "front/1", 0.05)
    validation_distance = KnownDistance("left/0", "left/1", 0.05)
    solver = JointZeroCalibrator(backend, residual_tolerance=0.0001)
    fit = solver.calibrate_groups(groups[:2], distances=[training_distance])
    nominal = solver.evaluate_groups(
        groups[2:], offsets_rad=np.zeros(6), distances=[validation_distance]
    )
    corrected = solver.evaluate_groups(
        groups[2:], offsets_rad=fit.offsets_rad, distances=[validation_distance]
    )
    validation_passed = all(g.point_rms_m <= 0.0001 for g in corrected.groups) and all(
        abs(d.error_m) <= 0.0001 for d in corrected.distances
    )
    payload = {
        "synthetic": True,
        "encoder_noise_std_deg": noise,
        "seed": 42,
        "joint_names": list(backend.joint_names),
        "sample_unit": "rad",
        "socket_distance_m": 0.05,
        "tcp_xyz_m": [0.08, 0, 0.16],
        "injected_offsets_deg": np.rad2deg(expected).tolist(),
        "offset_error_deg": np.rad2deg(fit.offsets_rad - expected).tolist(),
        "training": fit.to_dict(),
        "validation_nominal": nominal.to_dict(),
        "validation_corrected": corrected.to_dict(),
        "validation_passed": validation_passed,
        "tolerance_m": 0.0001,
    }
    return groups, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation-count", type=int, default=21)
    parser.add_argument("--encoder-noise-deg", type=float, default=0.002)
    parser.add_argument(
        "--output-dir", type=Path, help="export named radian CSVs and JSON"
    )
    args = parser.parse_args(argv)
    try:
        groups, report = run_demo(
            count=args.orientation_count, encoder_noise_deg=args.encoder_noise_deg
        )
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for group in groups:
                stream = io.StringIO()
                np.savetxt(
                    stream,
                    group.samples,
                    delimiter=",",
                    header=",".join(report["joint_names"]),
                    comments="",
                )
                write_text_atomic(
                    args.output_dir / f"{group.name.replace('/', '_')}_rad.csv",
                    stream.getvalue(),
                )
            write_text_atomic(
                args.output_dir / "two_socket_result.json",
                json.dumps(report, indent=2, allow_nan=False) + "\n",
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    fit = report["training"]
    nominal, corrected = report["validation_nominal"], report["validation_corrected"]
    print("UR10 synthetic two-socket calibration (50 mm spacing)")
    print(f"Encoder noise: {args.encoder_noise_deg:g} deg standard deviation")
    print(f"Training: {fit['message']}; rank {fit['rank']}/5")
    print(f"Estimated offsets (deg): {np.round(fit['offsets_deg'], 6).tolist()}")
    print(
        f"Held-out point RMS: {nominal['point_rms_m'] * 1000:.6f} -> {corrected['point_rms_m'] * 1000:.6f} mm"
    )
    before, after = (
        nominal["distances"][0]["error_m"],
        corrected["distances"][0]["error_m"],
    )
    print(f"Held-out distance error: {before * 1000:+.6f} -> {after * 1000:+.6f} mm")
    print(
        f"Validation: {'PASS' if report['validation_passed'] else 'FAIL'} (0.1 mm tolerance)"
    )
    return 0 if fit["converged"] and report["validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
