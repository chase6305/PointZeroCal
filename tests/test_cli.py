import json
import os

import numpy as np
import pytest

from point_zero_cal import _io
from point_zero_cal.cli import main


def cli_args(planar_urdf, samples):
    return [
        "--urdf",
        str(planar_urdf),
        "--end-frame",
        "tip_link",
        "--samples",
        str(samples),
        "--unit",
        "rad",
    ]


def load_strict_json(text):
    def reject_constant(value):
        raise AssertionError(f"non-standard JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def test_cli_exports_same_strict_json_as_stdout(planar_urdf, tmp_path, capsys):
    samples = tmp_path / "samples.csv"
    output = tmp_path / "result.json"
    expected = np.array([0.0, -0.025])
    measured = np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]]) - expected
    np.savetxt(samples, measured, delimiter=",")

    code = main(cli_args(planar_urdf, samples) + ["--output", str(output)])
    captured = capsys.readouterr()
    result = load_strict_json(captured.out)

    assert code == 0
    assert captured.err == ""
    assert output.read_text() == captured.out
    np.testing.assert_allclose(result["offsets_rad"], expected, atol=1e-8)
    diagnostics = result["sample_diagnostics"]
    assert len(diagnostics["corrected_point_errors_m"]) == 2
    assert diagnostics["max_corrected_point_error_m"] < 1e-7
    assert result["sample_csv_rows"] == [1, 2]
    assert (
        diagnostics["worst_sample_csv_row"]
        == (result["sample_csv_rows"][diagnostics["worst_sample_index"]])
    )


def test_cli_calibrates_reordered_header_and_reports_csv_rows(
    planar_urdf, tmp_path, capsys
):
    samples = tmp_path / "named.csv"
    expected = np.array([0.0, -0.025])
    measured = np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]]) - expected
    samples.write_text(
        f"\ufeff# session\njoint2,joint1\n{measured[0, 1]},{measured[0, 0]}\n"
        f"\n# second pose\n{measured[1, 1]},{measured[1, 0]}\n",
        encoding="utf-8",
    )

    assert main(cli_args(planar_urdf, samples) + ["--csv-header"]) == 0
    result = load_strict_json(capsys.readouterr().out)

    np.testing.assert_allclose(result["offsets_rad"], expected, atol=1e-8)
    assert result["joint_names"] == ["joint1", "joint2"]
    assert result["sample_csv_rows"] == [3, 6]
    diagnostics = result["sample_diagnostics"]
    assert (
        diagnostics["worst_sample_csv_row"]
        == (result["sample_csv_rows"][diagnostics["worst_sample_index"]])
    )


@pytest.mark.parametrize("pose", [[0.0, 0.0], [0.3, 0.7]])
def test_cli_rank_deficiency_is_failure_with_null_condition(
    planar_urdf, tmp_path, capsys, pose
):
    samples = tmp_path / "samples.csv"
    np.savetxt(samples, np.tile(pose, (41, 1)), delimiter=",")

    assert main(cli_args(planar_urdf, samples)) == 2
    result = load_strict_json(capsys.readouterr().out)
    assert not result["converged"]
    assert result["rank"] == 0
    assert result["condition_number"] is None
    assert result["corrected_point_rms"] == 0


@pytest.mark.parametrize(
    "contents, extra, message",
    [
        (None, [], "not found"),
        ("invalid,data\n0,0\n", [], "convert"),
        ("0,0\n", [], "shape"),
        ("0,nan\n0,0\n", [], "finite"),
        ("joint2,other\n0,0\n0,0\n", ["--csv-header"], "missing joint names"),
        ("0,0\n0,0\n", ["--fixed-joint", "missing"], "unknown joint"),
        ("0,0\n0,0\n", ["--max-iterations", "0"], "max_iterations"),
        ("0,0\n0,0\n", ["--damping", "nan"], "damping"),
        ("0,0\n0,0\n", ["--residual-tolerance", "inf"], "residual_tolerance"),
        ("0,0\n0,0\n", ["--step-tolerance", "0"], "step_tolerance"),
        ("0,0\n0,0\n", ["--rank-tolerance", "1"], "rank_tolerance"),
        ("0,0\n0,0\n", ["--tcp-xyz", "nan", "0", "0"], "tcp_xyz"),
        ("0,0\n0,0\n", ["--end-frame", "missing"], "not found"),
    ],
)
def test_cli_reports_input_errors_without_traceback(
    planar_urdf, tmp_path, capsys, contents, extra, message
):
    samples = tmp_path / "samples.csv"
    if contents is not None:
        samples.write_text(contents)

    with pytest.raises(SystemExit) as error:
        main(cli_args(planar_urdf, samples) + extra)

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_cli_write_failure_does_not_print_success_json(planar_urdf, tmp_path, capsys):
    samples = tmp_path / "samples.csv"
    np.savetxt(samples, [[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]], delimiter=",")

    with pytest.raises(SystemExit) as error:
        main(cli_args(planar_urdf, samples) + ["--output", str(tmp_path)])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot write output" in captured.err


def test_cli_residual_tolerance_controls_noisy_fit_acceptance(
    planar_urdf, tmp_path, capsys
):
    samples = tmp_path / "samples.csv"
    np.savetxt(
        samples, [[0.0, np.pi / 2], [np.pi / 2 + 0.01, -np.pi / 2]], delimiter=","
    )
    args = cli_args(planar_urdf, samples)

    assert main(args) == 2
    strict = load_strict_json(capsys.readouterr().out)
    assert main(args + ["--residual-tolerance", "0.01"]) == 0
    relaxed = load_strict_json(capsys.readouterr().out)
    assert relaxed["corrected_point_rms"] == pytest.approx(
        strict["corrected_point_rms"]
    )


@pytest.mark.parametrize("source_name", ["samples", "urdf"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_cli_does_not_overwrite_inputs_through_output_aliases(
    planar_urdf, tmp_path, capsys, source_name, alias
):
    samples = tmp_path / "samples.csv"
    np.savetxt(samples, [[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]], delimiter=",")
    original_samples = samples.read_bytes()
    original_urdf = planar_urdf.read_bytes()
    source = samples if source_name == "samples" else planar_urdf
    output = source if alias == "direct" else tmp_path / "output.json"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, output)

    with pytest.raises(SystemExit) as error:
        main(cli_args(planar_urdf, samples) + ["--output", str(output)])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "output must differ from input" in captured.err
    assert samples.read_bytes() == original_samples
    assert planar_urdf.read_bytes() == original_urdf


@pytest.mark.parametrize("failed_operation", ["fsync", "replace"])
def test_cli_preserves_existing_report_when_publication_fails(
    planar_urdf, tmp_path, capsys, monkeypatch, failed_operation
):
    samples = tmp_path / "samples.csv"
    np.savetxt(samples, [[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]], delimiter=",")
    report = tmp_path / "report.json"
    previous = '{"previous_calibration": true}\n'
    report.write_text(previous)

    def fail(*args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(_io.os, failed_operation, fail)
    with pytest.raises(SystemExit) as error:
        main(cli_args(planar_urdf, samples) + ["--output", str(report)])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot write output" in captured.err
    assert report.read_text() == previous
    assert not list(tmp_path.glob(".report.json.*.tmp"))
