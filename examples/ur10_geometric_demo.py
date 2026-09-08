"""Recover TCP mounting translation errors from two-socket contact data.

Deviations are injected into the fixed ``jointEE`` origin (the tool mounting)
of the UR10; the nominal model calibrates them from two sockets, the fitted
values are exported into a calibrated URDF copy, and an independent fixture
placement checks the improvement. Joint zero offsets and mounting rotations
are separate models: this demo fits geometric origin parameters only and
never estimates joint offsets at the same time.

Fixed-point consistency sees each parameter only through its non-rigid TCP
effect, so priors compete with the data through the Jacobian: generous
``prior_scale`` values let the contact data dominate, tight ones hold the
parameters near nominal. The report prints the data-only singular values and
both fits so the trade-off is visible.
"""

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from point_zero_cal import (
    ContactGroup,
    GeometricCalibrator,
    GeometricParameter,
    KnownDistance,
    PinocchioKinematics,
    export_calibrated_urdf,
    geometric_origin_updates,
)
from point_zero_cal._io import write_text_atomic
from point_zero_cal.geometric_backend import PinocchioGeometricKinematics
from point_zero_cal.simulation import generate_fixed_point_dataset, make_rpy_samples

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/UR10/UR10.urdf"
TCP = (0.08, 0, 0.16)
INJECTED = {"x": 0.003, "y": -0.004, "z": 0.005}
SEED = np.array([0, -1.2, 1.8, -1.4, -1.57, 0])


def write_perturbed_urdf(destination: Path) -> Path:
    """Write a copy of the UR10 URDF with injected jointEE deviations."""
    tree = ET.parse(URDF)
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") != "jointEE":
            continue
        origin = joint.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz").split()])
        for component, value in INJECTED.items():
            xyz["xyz".index(component)] += value
        origin.set("xyz", " ".join(repr(float(v)) for v in xyz))
    tree.write(destination)
    return destination


def run_demo(*, count: int = 21, prior_scale: float = 10.0):
    """Return synthetic groups and the geometric fit/validation reports."""
    true_backend = None
    temporary = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix="point-zero-cal-geometric-"))
        perturbed = write_perturbed_urdf(temporary / "perturbed.urdf")
        true_backend = PinocchioKinematics(perturbed, "ee_link", tcp_xyz=TCP)
        parameters = [
            GeometricParameter("jointEE", component, prior_scale)
            for component in ("x", "y", "z")
        ]
        nominal = PinocchioGeometricKinematics(URDF, "ee_link", parameters, tcp_xyz=TCP)
        origin = np.array(true_backend.tcp_pose(SEED).translation)
        deltas = make_rpy_samples(count=count, max_angle_deg=40)
        placements = {
            "front/0": [0, 0, 0],
            "front/1": [0.05, 0, 0],
            "left/0": [0, 0.08, 0.03],
            "left/1": [0.05, 0.08, 0.03],
        }
        groups = []
        for name, shift in placements.items():
            dataset = generate_fixed_point_dataset(
                true_backend, SEED, deltas, target_xyz=origin + shift
            )
            groups.append(ContactGroup(name, dataset.configurations_rad))
        solver = GeometricCalibrator(nominal, residual_tolerance=0.001)
        fit = solver.calibrate_groups(
            groups[:2], distances=[KnownDistance("front/0", "front/1", 0.05)]
        )
        estimates = [parameter.estimate for parameter in fit.parameters]
        nominal_report = solver.evaluate_groups(
            groups[2:],
            parameters=[0.0, 0.0, 0.0],
            distances=[KnownDistance("left/0", "left/1", 0.05)],
        )
        corrected_report = solver.evaluate_groups(
            groups[2:],
            parameters=estimates,
            distances=[KnownDistance("left/0", "left/1", 0.05)],
        )
        tolerance = 0.0005
        validation_passed = all(
            group.point_rms_m <= tolerance for group in corrected_report.groups
        ) and all(
            abs(distance.error_m) <= tolerance
            for distance in corrected_report.distances
        )
        # A tight-prior fit shows how strongly the nominal model is trusted.
        tight_backend = PinocchioGeometricKinematics(
            URDF,
            "ee_link",
            [
                GeometricParameter("jointEE", component, 0.002)
                for component in ("x", "y", "z")
            ],
            tcp_xyz=TCP,
        )
        tight = GeometricCalibrator(
            tight_backend, residual_tolerance=0.001
        ).calibrate_groups(
            groups[:2], distances=[KnownDistance("front/0", "front/1", 0.05)]
        )
        payload = {
            "synthetic": True,
            "seed": 42,
            "joint_names": list(nominal.joint_names),
            "sample_unit": "rad",
            "tcp_xyz_m": list(TCP),
            "socket_distance_m": 0.05,
            "injected_deviations_m": INJECTED,
            "prior_scale": prior_scale,
            "data_singular_values": fit.singular_values.tolist(),
            "fit": fit.to_dict(),
            "estimates_m": {
                component: estimate
                for component, estimate in zip(("x", "y", "z"), estimates, strict=True)
            },
            "tight_prior_estimates_m": {
                component: parameter.estimate
                for component, parameter in zip(
                    ("x", "y", "z"), tight.parameters, strict=True
                )
            },
            "validation_nominal": nominal_report.to_dict(),
            "validation_corrected": corrected_report.to_dict(),
            "validation_passed": validation_passed,
            "validation_tolerance_m": tolerance,
        }
        return groups, payload, fit, nominal, estimates
    finally:
        if true_backend is not None and temporary is not None:
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation-count", type=int, default=21)
    parser.add_argument("--prior-scale", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="export the calibrated URDF and result JSON",
    )
    args = parser.parse_args(argv)
    _, payload, fit, _, _ = run_demo(
        count=args.orientation_count, prior_scale=args.prior_scale
    )
    injected = payload["injected_deviations_m"]
    print(f"Data-only singular values: {payload['data_singular_values']}")
    print(f"Fit: {fit.message} (data rank {fit.data_rank}/{len(fit.parameters)})")
    for component, true in injected.items():
        estimate = payload["estimates_m"][component]
        tight = payload["tight_prior_estimates_m"][component]
        print(
            f"  jointEE.{component}: injected {true:+.4f} m | "
            f"estimate {estimate:+.4f} m | tight prior {tight:+.4f} m"
        )
    nominal_rms = payload["validation_nominal"]["point_rms_m"] * 1e3
    corrected_rms = payload["validation_corrected"]["point_rms_m"] * 1e3
    print(f"Held-out point RMS: {nominal_rms:.3f} -> {corrected_rms:.3f} mm")
    print(f"Validation: {'PASS' if payload['validation_passed'] else 'FAIL'}")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        export_calibrated_urdf(
            URDF,
            args.output_dir / "ur10_calibrated.urdf",
            geometric_origin_updates(URDF, fit.parameters),
            metadata={
                "point-zero-cal": True,
                "parameters": [parameter.to_dict() for parameter in fit.parameters],
                "converged": fit.converged,
                "message": fit.message,
                "validation_passed": payload["validation_passed"],
            },
        )
        write_text_atomic(
            args.output_dir / "ur10_geometric_result.json",
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        print(f"Exported calibrated URDF and JSON to {args.output_dir}")
    return 0 if fit.converged and payload["validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
