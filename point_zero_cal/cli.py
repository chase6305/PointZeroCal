"""Command-line interface for PointZeroCal."""

import argparse
import json
from pathlib import Path

from ._io import validate_output_path, write_text_atomic
from .calibrator import JointZeroCalibrator
from .contacts import ContactGroup, KnownDistance
from .pinocchio_backend import PinocchioKinematics
from .samples import load_samples_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(description="Calibrate robot joint zero offsets")
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--end-frame", required=True)
    parser.add_argument(
        "--samples",
        type=Path,
        help="CSV with one pose per row (single-contact mode)",
    )
    parser.add_argument(
        "--csv-header",
        action="store_true",
        help="read joint names from the first data row and reorder columns to model order",
    )
    parser.add_argument("--unit", choices=("deg", "rad"), default="deg")
    parser.add_argument(
        "--tcp-xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--tcp-rpy",
        nargs=3,
        type=float,
        metavar=("R", "P", "Y"),
        default=(0.0, 0.0, 0.0),
        help="TCP rotation in radians relative to end-frame",
    )
    parser.add_argument(
        "--fixed-joint",
        action="append",
        help="joint name to hold fixed (defaults to the first actuated joint)",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--residual-tolerance",
        type=float,
        default=1e-6,
        help="maximum corrected point RMS in metres (default: 1e-6)",
    )
    parser.add_argument(
        "--step-tolerance",
        type=float,
        default=1e-8,
        help="remaining undamped offset correction tolerance in radians (default: 1e-8)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=1e-10,
        help="non-negative least-squares regularization (default: 1e-10)",
    )
    parser.add_argument(
        "--rank-tolerance",
        type=float,
        default=1e-8,
        help="relative singular-value cutoff between 0 and 1 (default: 1e-8)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--group",
        action="append",
        metavar="NAME=PATH",
        help="named contact CSV in multi-contact mode (repeatable)",
    )
    parser.add_argument(
        "--group-sigma",
        action="append",
        metavar="NAME=SIGMA",
        help="positional noise sigma in metres for a named group",
    )
    parser.add_argument(
        "--validate-group",
        action="append",
        metavar="NAME=PATH",
        help="held-out contact CSV evaluated at the fitted offsets",
    )
    parser.add_argument(
        "--distance",
        action="append",
        nargs=3,
        metavar=("FIRST", "SECOND", "METERS"),
        help="known centre distance between two contact groups",
    )
    parser.add_argument(
        "--distance-weight",
        action="append",
        nargs=3,
        metavar=("FIRST", "SECOND", "WEIGHT"),
        help="positive dimensionless weight for one known distance",
    )
    parser.add_argument(
        "--distance-sigma",
        action="append",
        nargs=3,
        metavar=("FIRST", "SECOND", "SIGMA"),
        help="measurement uncertainty in metres for one known distance",
    )
    parser.add_argument(
        "--validate-distance",
        action="append",
        nargs=3,
        metavar=("FIRST", "SECOND", "METERS"),
        help="known distance between two validation groups",
    )
    parser.add_argument(
        "--validate-distance-sigma",
        action="append",
        nargs=3,
        metavar=("FIRST", "SECOND", "SIGMA"),
        help="measurement uncertainty for one validation distance",
    )
    return parser


def _name_value(entries: list[str], what: str) -> dict[str, str]:
    """Parse NAME=VALUE entries, rejecting empty names and duplicates."""
    parsed = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"--{what} requires NAME=VALUE, got {entry!r}")
        name, value = entry.split("=", 1)
        if not name.strip():
            raise ValueError(f"--{what} requires a non-empty group name")
        if not value.strip():
            raise ValueError(f"--{what} requires a non-empty value")
        if name in parsed:
            raise ValueError(f"duplicate --{what} entry for {name!r}")
        parsed[name] = value
    return parsed


def _pair_entries(
    entries: list[list[str]], names: set[str], what: str
) -> dict[frozenset[str], str]:
    parsed = {}
    for first, second, value in entries or []:
        if first == second:
            raise ValueError(f"--{what} endpoints must be different groups")
        if first not in names or second not in names:
            raise ValueError(
                f"--{what} references an unknown contact group: {first!r} or {second!r}"
            )
        key = frozenset((first, second))
        if key in parsed:
            raise ValueError(f"duplicate --{what} entry for {first!r} and {second!r}")
        parsed[key] = value
    return parsed


def _parse_sigma(value: str, what: str) -> float:
    try:
        sigma = float(value)
    except ValueError as exc:
        raise ValueError(f"{what}: invalid sigma {value!r}") from exc
    return sigma


def _build_known_distances(
    measured_entries: list[list[str]],
    weight_entries: list[list[str]],
    sigma_entries: list[list[str]],
    names: set[str],
    what: str,
) -> list[KnownDistance]:
    """Resolve distance flags into KnownDistance measurements."""
    measured = _pair_entries(measured_entries, names, what)
    weights = _pair_entries(weight_entries, names, f"{what}-weight")
    sigmas = _pair_entries(sigma_entries, names, f"{what}-sigma")
    for suffix, overrides in (("weight", weights), ("sigma", sigmas)):
        if overrides.keys() - measured.keys():
            raise ValueError(
                f"--{what}-{suffix} requires a matching --{what} measurement"
            )
    distances = []
    for key, metres_text in measured.items():
        try:
            metres = float(metres_text)
        except ValueError as exc:
            raise ValueError(f"{what}: invalid distance {metres_text!r}") from exc
        weight = 1.0
        if key in weights:
            try:
                weight = float(weights[key])
            except ValueError as exc:
                raise ValueError(
                    f"{what}-weight: invalid weight {weights[key]!r}"
                ) from exc
        sigma = None
        if key in sigmas:
            sigma = _parse_sigma(sigmas[key], f"{what}-sigma")
        first, second = sorted(key)
        distances.append(KnownDistance(first, second, metres, weight, sigma))
    return distances


def _run_groups(
    args: argparse.Namespace, backend: PinocchioKinematics
) -> tuple[dict[str, object], bool, bool | None]:
    """Run multi-contact calibration; raise ValueError on invalid inputs."""
    group_paths = _name_value(args.group or [], "group")
    validate_paths = _name_value(args.validate_group or [], "validate-group")
    if not group_paths and not validate_paths:
        raise ValueError("group mode requires at least one --group or --validate-group")
    overlap = set(group_paths) & set(validate_paths)
    if overlap:
        raise ValueError(f"training and validation group names must differ: {overlap}")
    group_sigmas = _name_value(args.group_sigma or [], "group-sigma")
    for name in group_sigmas:
        if name not in group_paths and name not in validate_paths:
            raise ValueError(f"--group-sigma references an unknown group: {name!r}")
    distances = _build_known_distances(
        args.distance or [],
        args.distance_weight or [],
        args.distance_sigma or [],
        set(group_paths),
        "distance",
    )
    validation_distances = _build_known_distances(
        args.validate_distance or [],
        [],
        args.validate_distance_sigma or [],
        set(validate_paths),
        "validate-distance",
    )
    tables = {
        name: load_samples_csv(Path(path), backend.joint_names, header=args.csv_header)
        for name, path in {**group_paths, **validate_paths}.items()
    }
    groups = [
        ContactGroup(
            name,
            tables[name].values,
            csv_rows=tables[name].csv_rows,
            sigma_m=(
                _parse_sigma(group_sigmas[name], "group-sigma")
                if name in group_sigmas
                else None
            ),
        )
        for name in group_paths
    ]
    fixed_joints = args.fixed_joint if args.fixed_joint is not None else (0,)
    solver = JointZeroCalibrator(
        backend,
        max_iterations=args.max_iterations,
        residual_tolerance=args.residual_tolerance,
        step_tolerance=args.step_tolerance,
        damping=args.damping,
        rank_tolerance=args.rank_tolerance,
    )
    result = solver.calibrate_groups(
        groups,
        distances=distances,
        sample_unit=args.unit,
        fixed_joints=fixed_joints,
    )
    payload = result.to_dict()
    payload["joint_names"] = list(backend.joint_names)
    payload["group_csv_rows"] = {
        name: list(tables[name].csv_rows) for name in group_paths
    }
    validation = None
    validation_passed = None
    if validate_paths:
        validation_groups = [
            ContactGroup(
                name,
                tables[name].values,
                csv_rows=tables[name].csv_rows,
                sigma_m=(
                    _parse_sigma(group_sigmas[name], "group-sigma")
                    if name in group_sigmas
                    else None
                ),
            )
            for name in validate_paths
        ]
        validation = solver.evaluate_groups(
            validation_groups,
            offsets_rad=result.offsets_rad,
            distances=validation_distances,
            sample_unit=args.unit,
        )
        validation_passed = all(
            group.point_rms_m <= args.residual_tolerance for group in validation.groups
        ) and all(
            abs(distance.error_m) <= args.residual_tolerance
            for distance in validation.distances
        )
        payload["validation_evaluation"] = validation.to_dict()
        payload["validation_passed"] = validation_passed
    return payload, result.converged, validation_passed


def main(argv: list[str] | None = None) -> int:
    """Run calibration and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    group_mode = bool(args.group or args.validate_group)
    try:
        if group_mode and args.samples:
            parser.error("--samples cannot be combined with --group mode")
        if not group_mode and not args.samples:
            parser.error("either --samples or --group is required")
        if group_mode and not args.group:
            parser.error("--validate-group requires at least one --group")
        if not group_mode:
            for option in (
                "group_sigma", "distance", "distance_weight", "distance_sigma",
                "validate_distance", "validate_distance_sigma",
            ):
                if getattr(args, option):
                    parser.error(f"--{option.replace('_', '-')} requires --group mode")
        if args.output:
            inputs = [args.urdf]
            if args.samples:
                inputs.append(args.samples)
            if group_mode:
                for entries, option in (
                    (args.group, "group"), (args.validate_group, "validate-group")
                ):
                    inputs.extend(
                        Path(path) for path in _name_value(entries or [], option).values()
                    )
            validate_output_path(args.output, *inputs)
        backend = PinocchioKinematics(
            args.urdf,
            args.end_frame,
            tcp_xyz=tuple(args.tcp_xyz),
            tcp_rpy=tuple(args.tcp_rpy),
        )
        if group_mode:
            payload, converged, validation_passed = _run_groups(args, backend)
            text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                write_text_atomic(args.output, text + "\n")
            print(text)
            return 0 if converged and validation_passed is not False else 2
        table = load_samples_csv(
            args.samples, backend.joint_names, header=args.csv_header
        )
        fixed_joints = args.fixed_joint if args.fixed_joint is not None else (0,)
        result = JointZeroCalibrator(
            backend,
            max_iterations=args.max_iterations,
            residual_tolerance=args.residual_tolerance,
            step_tolerance=args.step_tolerance,
            damping=args.damping,
            rank_tolerance=args.rank_tolerance,
        ).calibrate(table.values, sample_unit=args.unit, fixed_joints=fixed_joints)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    try:
        payload = result.to_dict()
        payload["joint_names"] = list(backend.joint_names)
        payload["sample_csv_rows"] = list(table.csv_rows)
        if result.sample_diagnostics is not None:
            payload["sample_diagnostics"]["worst_sample_csv_row"] = table.csv_rows[
                result.sample_diagnostics.worst_sample_index
            ]
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            write_text_atomic(args.output, text + "\n")
    except OSError as exc:
        parser.error(f"cannot write output: {exc}")
    except ValueError as exc:
        parser.error(f"cannot serialize result: {exc}")
    print(text)
    return 0 if result.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
