"""Compare axial and eccentric UR10 TCP calibration cases in Viser."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from point_zero_cal import CalibrationResult, JointZeroCalibrator, PinocchioKinematics
from point_zero_cal.simulation import (
    FixedPointDataset,
    generate_fixed_point_dataset,
    make_rpy_samples,
)


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/UR10/UR10.urdf"
SEED = np.array([0.0, -1.2, 1.8, -1.4, -1.57, 0.0])


@dataclass(frozen=True)
class DemoCase:
    """Inputs and outputs needed to report one calibration scenario."""

    name: str
    tcp_xyz: tuple[float, float, float]
    joint_names: tuple[str, ...]
    fixed_joints: tuple[str, ...]
    dataset: FixedPointDataset
    injected_deg: np.ndarray
    result: CalibrationResult
    measured_rad: np.ndarray
    raw_point_rms: float
    corrected_point_rms: float


def point_rms(backend: PinocchioKinematics, configurations: np.ndarray) -> float:
    positions = np.vstack(
        [backend.position_and_jacobian(configuration)[0] for configuration in configurations]
    )
    center = positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))


def build_cases(rpy_deltas: np.ndarray) -> dict[str, DemoCase]:
    cases: dict[str, DemoCase] = {}
    settings = {
        # ee_link Z is collinear with Joint6: fix Joint1 and Joint6.
        "Axial TCP - Joint6 unobservable": (
            (0.0, 0.0, 0.16),
            [0.0, 1.20, -0.80, 1.50, -1.00, 2.00],
            ("Joint1", "Joint6"),
        ),
        # The X component gives Joint6 a translational lever arm.
        "Lateral X-offset TCP - Joint6 observable": (
            (0.08, 0.0, 0.16),
            [0.0, 1.20, -0.80, 1.50, -1.00, 2.00],
            ("Joint1",),
        ),
    }
    for name, (tcp_xyz, injected, fixed) in settings.items():
        backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=tcp_xyz)
        dataset = generate_fixed_point_dataset(backend, SEED, rpy_deltas)
        injected_deg = np.asarray(injected)
        measured = dataset.configurations_rad - np.deg2rad(injected_deg)
        result = JointZeroCalibrator(backend).calibrate(measured, fixed_joints=fixed)
        corrected = measured + result.offsets_rad
        cases[name] = DemoCase(
            name,
            tcp_xyz,
            backend.joint_names,
            fixed,
            dataset,
            injected_deg,
            result,
            measured,
            point_rms(backend, measured),
            point_rms(backend, corrected),
        )
    return cases


def print_cases(cases: dict[str, DemoCase]) -> None:
    """Print a compact, human-readable acceptance report."""
    print("\n" + "=" * 78)
    print(" PointZeroCal | UR10 fixed-point calibration validation")
    print("=" * 78)
    for case_number, case in enumerate(cases.values(), start=1):
        print(f"\nCASE {case_number}: {case.name}")
        print("-" * 78)
        target = case.dataset.target_xyz
        print(
            "Setup       : "
            f"TCP xyz={case.tcp_xyz} m | samples={len(case.dataset.configurations_rad)}"
        )
        print(f"Target point: [{target[0]:+.6f}, {target[1]:+.6f}, {target[2]:+.6f}] m")
        print(f"IK max error: {case.dataset.position_errors.max():.3e} m")
        joint_span_deg = np.ptp(np.rad2deg(case.dataset.configurations_rad), axis=0)
        spans = ", ".join(
            f"{name}={span:.1f} deg"
            for name, span in zip(case.joint_names, joint_span_deg)
        )
        print(f"Joint spans : {spans}")

        print("\nJoint offset comparison")
        print(f"{'Joint':<10}{'Injected':>12}{'Estimated':>12}{'Error':>12}  {'State'}")
        print(f"{'':<10}{'(deg)':>12}{'(deg)':>12}{'(deg)':>12}")
        print("-" * 62)
        errors = case.result.offsets_deg - case.injected_deg
        for name, injected, estimated, error in zip(
            case.joint_names, case.injected_deg, case.result.offsets_deg, errors
        ):
            if name in case.result.active_joint_names:
                state = "CALIBRATED"
            elif name == "Joint1":
                state = "FIXED / REFERENCE"
            else:
                state = "FIXED / UNOBSERVABLE"
            print(
                f"{name:<10}{injected:>+12.6f}{estimated:>+12.6f}"
                f"{error:>+12.3e}  {state}"
            )

        improvement = (
            case.raw_point_rms / case.corrected_point_rms
            if case.corrected_point_rms > 0
            else float("inf")
        )
        expected_rank = len(case.result.active_joint_names)
        status = "PASS" if case.result.converged else "FAIL"
        print("\nCalibration quality")
        print(f"Status       : {status} - {case.result.message}")
        print(f"Iterations   : {case.result.iterations}")
        print(f"Point RMS    : {case.raw_point_rms * 1e3:.6f} mm -> "
              f"{case.corrected_point_rms * 1e3:.3e} mm")
        print(f"Improvement  : {improvement:.3e}x")
        print(f"Rank         : {case.result.rank}/{expected_rank}")
        print(f"Condition no.: {case.result.condition_number:.6f}")
    print("\n" + "=" * 78)


def export_cases(cases: dict[str, DemoCase], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases.values(), start=1):
        stem = "axial_tcp" if index == 1 else "eccentric_tcp"
        np.savetxt(output_dir / f"{stem}_measured_deg.csv", np.rad2deg(case.measured_rad), delimiter=",")
        payload = {
            "case": case.name,
            "tcp_xyz_m": case.tcp_xyz,
            "sample_count": len(case.measured_rad),
            "injected_offsets_deg": case.injected_deg.tolist(),
            "estimated_offsets_deg": case.result.offsets_deg.tolist(),
            "offset_error_deg": (case.result.offsets_deg - case.injected_deg).tolist(),
            "raw_point_rms_m": case.raw_point_rms,
            "corrected_point_rms_m": case.corrected_point_rms,
            "converged": case.result.converged,
            "message": case.result.message,
            "iterations": case.result.iterations,
            "active_joint_names": case.result.active_joint_names,
            "rank": case.result.rank,
            "condition_number": case.result.condition_number,
            "singular_values": case.result.singular_values.tolist(),
        }
        (output_dir / f"{stem}_result.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--orientation-range-deg", type=float, default=60.0)
    parser.add_argument("--orientation-count", type=int, default=41)
    parser.add_argument("--output-dir", type=Path, help="export measured CSV and result JSON")
    args = parser.parse_args()
    rpy_deltas = make_rpy_samples(
        max_angle_deg=args.orientation_range_deg, count=args.orientation_count
    )
    cases = build_cases(rpy_deltas)
    print_cases(cases)
    if args.output_dir:
        export_cases(cases, args.output_dir)
        print(f"Output       : exported calibration data to {args.output_dir}")
    if args.no_viewer:
        return

    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise SystemExit("Install the viewer with `pip install -e '.[simulation]'`") from exc
    server = viser.ViserServer()
    robot = ViserUrdf(server, URDF)
    first = next(iter(cases.values()))
    point = server.scene.add_icosphere(
        "/fixed_tcp_point", radius=0.025, color=(255, 40, 40), position=first.dataset.target_xyz
    )
    selector = server.gui.add_dropdown("TCP case", options=tuple(cases), initial_value=first.name)
    slider = server.gui.add_slider(
        "Contact pose", min=0, max=len(rpy_deltas) - 1, step=1, initial_value=0
    )

    def update() -> None:
        case = cases[selector.value]
        robot.update_cfg(case.dataset.configurations_rad[int(slider.value)])
        point.position = case.dataset.target_xyz

    selector.on_update(lambda _: update())
    slider.on_update(lambda _: update())
    update()
    print("Viewer       : Viser is running; switch TCP case and contact pose in the browser.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
