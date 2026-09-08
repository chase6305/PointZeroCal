"""Multi-contact CLI mode: groups, distances and held-out validation."""

import json

import numpy as np
import pytest

from point_zero_cal.cli import main as cli_main

DISTANCE = float(np.hypot(np.sqrt(3) - 1, 1))


def write_csv(path, values):
    np.savetxt(path, values, delimiter=",")
    return path


@pytest.fixture
def group_argv(planar_urdf, tmp_path):
    """CLI arguments for the two-group planar fit plus one validation group."""
    true = np.array([0.0, -0.025])
    front0 = write_csv(
        tmp_path / "front0.csv",
        np.array([[0, np.pi / 2], [np.pi / 2, -np.pi / 2]]) - true,
    )
    front1 = write_csv(
        tmp_path / "front1.csv",
        np.array([[-np.pi / 6, np.pi / 3], [np.pi / 6, -np.pi / 3]]) - true,
    )
    left0 = write_csv(
        tmp_path / "left0.csv",
        np.array([[0, 2 * np.pi / 3], [2 * np.pi / 3, -2 * np.pi / 3]]) - true,
    )
    return planar_urdf, front0, front1, left0


def run_group_cli(planar_urdf, *group_args, capsys):
    argv = [
        "--urdf",
        str(planar_urdf),
        "--end-frame",
        "tip_link",
        "--unit",
        "rad",
        *group_args,
    ]
    code = cli_main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_group_mode_fits_and_validates_held_out_contacts(group_argv, capsys):
    planar_urdf, front0, front1, left0 = group_argv
    code, payload = run_group_cli(
        planar_urdf,
        "--group",
        f"front0={front0}",
        "--group",
        f"front1={front1}",
        "--distance",
        "front0",
        "front1",
        str(DISTANCE),
        "--validate-group",
        f"left0={left0}",
        capsys=capsys,
    )
    assert code == 0
    assert payload["converged"] is True
    np.testing.assert_allclose(payload["offsets_rad"], [0, -0.025], atol=1e-8)
    assert payload["group_csv_rows"] == {"front0": [1, 2], "front1": [1, 2]}
    validation = payload["validation_evaluation"]
    assert validation["groups"][0]["name"] == "left0"
    assert payload["validation_passed"] is True
    assert validation["groups"][0]["point_rms_m"] < 1e-8


def test_group_mode_with_sigmas_reports_and_accepts(group_argv, capsys):
    planar_urdf, front0, front1, _ = group_argv
    code, payload = run_group_cli(
        planar_urdf,
        "--group",
        f"front0={front0}",
        "--group",
        f"front1={front1}",
        "--group-sigma",
        "front0=0.001",
        "--distance",
        "front0",
        "front1",
        str(DISTANCE),
        "--distance-sigma",
        "front0",
        "front1",
        "0.003",
        capsys=capsys,
    )
    assert code == 0
    assert payload["evaluation"]["groups"][0]["sigma_m"] == 0.001
    assert payload["evaluation"]["distances"][0]["sigma_m"] == 0.003


def test_group_mode_exports_strict_json_to_output(group_argv, tmp_path, capsys):
    planar_urdf, front0, front1, _ = group_argv
    output = tmp_path / "result.json"
    code, _ = run_group_cli(
        planar_urdf,
        "--group",
        f"front0={front0}",
        "--group",
        f"front1={front1}",
        "--output",
        str(output),
        capsys=capsys,
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["rank"] == 1
    np.testing.assert_allclose(payload["offsets_rad"], [0, -0.025], atol=1e-8)


def test_failed_validation_forces_exit_code_two(group_argv, tmp_path, capsys):
    planar_urdf, front0, front1, _ = group_argv
    bad = write_csv(
        tmp_path / "bad.csv",
        np.array([[0, 2 * np.pi / 3 + 0.05], [2 * np.pi / 3, -2 * np.pi / 3]]),
    )
    code, payload = run_group_cli(
        planar_urdf,
        "--group",
        f"front0={front0}",
        "--group",
        f"front1={front1}",
        "--distance",
        "front0",
        "front1",
        str(DISTANCE),
        "--validate-group",
        f"bad={bad}",
        capsys=capsys,
    )
    assert code == 2
    assert payload["converged"] is True  # the fit itself succeeds
    assert payload["validation_passed"] is False
    assert payload["validation_evaluation"]["groups"][0]["point_rms_m"] > 0.005


@pytest.mark.parametrize(
    "args, match",
    [
        ([], "either --samples or --group"),
        (["--samples", "x.csv", "--group", "a=x.csv"], "cannot be combined"),
        (["--group", "noequals"], "requires NAME=VALUE"),
        (["--group", "a=x", "--group", "a=y"], "duplicate"),
        (["--group", "a=x", "--group-sigma", "missing=0.1"], "unknown group"),
        (["--group", "a=x", "--group-sigma", "a=nan"], "sigma"),
        (
            ["--group", "a=x", "--distance", "a", "missing", "0.05"],
            "unknown contact group",
        ),
        (
            ["--group", "a=x", "--distance", "a", "a", "0.05"],
            "different groups",
        ),
        (
            ["--group", "a=x", "--group", "b=y", "--distance", "a", "b", "notanumber"],
            "invalid distance",
        ),
    ],
)
def test_group_mode_rejects_invalid_invocations(planar_urdf, capsys, args, match):
    with pytest.raises(SystemExit) as excinfo:
        cli_main(
            [
                "--urdf",
                str(planar_urdf),
                "--end-frame",
                "tip_link",
                *args,
            ]
        )
    assert excinfo.value.code == 2
    assert match in capsys.readouterr().err


@pytest.mark.parametrize("extra, message", [
    (["--group", "missing-equals"], "requires NAME=VALUE"),
    (["--group", "a="], "non-empty value"),
    (["--validate-group", "a=data.csv"], "requires at least one --group"),
    (["--group", "a=x", "--group", "b=y", "--distance-weight", "a", "b", "2"],
     "requires a matching --distance measurement"),
    (["--group", "a=x", "--group", "b=y", "--distance-sigma", "a", "b", "0.01"],
     "requires a matching --distance measurement"),
])
def test_invalid_group_options_with_output_preserve_file(
    planar_urdf, tmp_path, capsys, extra, message
):
    output = tmp_path / "result.json"
    output.write_text("previous result")
    with pytest.raises(SystemExit) as excinfo:
        cli_main([
            "--urdf", str(planar_urdf), "--end-frame", "tip_link",
            "--output", str(output), *extra,
        ])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err
    assert output.read_text() == "previous result"


@pytest.mark.parametrize("extra", [
    ["--group-sigma", "a=0.01"],
    ["--distance", "a", "b", "0.1"],
    ["--distance-weight", "a", "b", "2"],
    ["--distance-sigma", "a", "b", "0.01"],
    ["--validate-distance", "a", "b", "0.1"],
    ["--validate-distance-sigma", "a", "b", "0.01"],
])
def test_single_contact_mode_rejects_group_only_flags(group_argv, capsys, extra):
    urdf, samples, _, _ = group_argv
    with pytest.raises(SystemExit) as excinfo:
        cli_main([
            "--urdf", str(urdf), "--end-frame", "tip_link",
            "--samples", str(samples), "--unit", "rad", *extra,
        ])
    assert excinfo.value.code == 2
    assert f"{extra[0]} requires --group mode" in capsys.readouterr().err
