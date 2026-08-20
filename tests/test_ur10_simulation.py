import json
from pathlib import Path

import numpy as np
import pytest

from point_zero_cal import JointZeroCalibrator, PinocchioKinematics
from point_zero_cal.cli import main as cli_main
from point_zero_cal.simulation import (
    generate_fixed_point_dataset,
    make_rpy_grid,
    make_rpy_samples,
)


URDF = Path(__file__).parents[1] / "assets/UR10/UR10.urdf"


def dataset_for(backend):
    seed = np.array([0.0, -1.2, 1.8, -1.4, -1.57, 0.0])
    rpy_deltas = make_rpy_grid(max_angle_deg=12.0, levels=3)
    return generate_fixed_point_dataset(backend, seed, rpy_deltas)


def test_ur10_lateral_x_offset_tcp_recovers_joint6_offset():
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))
    dataset = dataset_for(backend)
    expected = np.deg2rad([0.0, 1.20, -0.80, 1.50, -1.00, 2.00])

    result = JointZeroCalibrator(backend).calibrate(
        dataset.configurations_rad - expected
    )

    assert dataset.position_errors.max() < 1e-8
    assert result.converged
    assert result.rank == 5
    np.testing.assert_allclose(result.offsets_rad, expected, atol=1e-7)


def test_lateral_tcp_makes_joint6_translational_jacobian_observable():
    """A lateral lever arm, unlike an axial extension, moves under Joint6."""
    q = np.array([0.0, -1.2, 1.8, -1.4, -1.57, 0.0])
    axial = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.0, 0.0, 0.16))
    lateral = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))

    axial_joint6 = np.linalg.norm(axial.position_and_jacobian(q)[1][:, 5])
    lateral_joint6 = np.linalg.norm(lateral.position_and_jacobian(q)[1][:, 5])

    assert axial_joint6 < 1e-12
    assert lateral_joint6 > 0.05


def test_default_rpy_grid_contains_125_symmetric_samples():
    grid = make_rpy_grid()
    assert grid.shape == (125, 3)
    np.testing.assert_allclose(grid.mean(axis=0), 0.0, atol=1e-15)
    assert np.unique(grid, axis=0).shape[0] == 125


def test_default_rpy_samples_contains_41_symmetric_samples():
    samples = make_rpy_samples()
    assert samples.shape == (41, 3)
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(samples[1::2], -samples[2::2], atol=1e-15)
    assert np.rad2deg(np.abs(samples)).max() > 50.0


def test_ur10_default_samples_are_distributed_in_joint_space():
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))
    dataset = generate_fixed_point_dataset(
        backend,
        np.array([0.0, -1.2, 1.8, -1.4, -1.57, 0.0]),
        make_rpy_samples(),
    )
    spans_deg = np.ptp(np.rad2deg(dataset.configurations_rad), axis=0)
    assert np.all(spans_deg > 30.0)
    assert spans_deg[3:].min() > 90.0


def test_ur10_axial_tcp_requires_joint6_to_be_fixed():
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.0, 0.0, 0.16))
    dataset = dataset_for(backend)
    injected = np.deg2rad([0.0, 1.20, -0.80, 1.50, -1.00, 2.00])
    observable = injected.copy()
    observable[-1] = 0.0
    measured = dataset.configurations_rad - injected

    underconstrained = JointZeroCalibrator(backend, max_iterations=5).calibrate(
        measured, fixed_joints=["Joint1"]
    )
    result = JointZeroCalibrator(backend).calibrate(
        measured, fixed_joints=["Joint1", "Joint6"]
    )

    assert not underconstrained.converged
    assert underconstrained.rank == 4
    assert result.converged
    assert result.rank == 4
    np.testing.assert_allclose(result.offsets_rad, observable, atol=1e-7)


def test_dataset_generator_rejects_empty_orientation_set():
    backend = PinocchioKinematics(URDF, "ee_link")
    with pytest.raises(ValueError, match="N >= 1"):
        generate_fixed_point_dataset(backend, np.zeros(6), [])


def test_cli_prints_strict_json_diagnostics(tmp_path, capsys):
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))
    dataset = dataset_for(backend)
    expected = np.deg2rad([0.0, 1.20, -0.80, 1.50, -1.00, 2.00])
    samples = tmp_path / "samples.csv"
    np.savetxt(samples, np.rad2deg(dataset.configurations_rad - expected), delimiter=",")

    exit_code = cli_main(
        [
            "--urdf", str(URDF),
            "--end-frame", "ee_link",
            "--tcp-xyz", "0.08", "0", "0.16",
            "--samples", str(samples),
            "--unit", "deg",
            "--fixed-joint", "Joint1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["message"] == "calibration converged"
    assert payload["sample_count"] == len(dataset.configurations_rad)
    assert payload["active_joint_names"][-1] == "Joint6"
    np.testing.assert_allclose(payload["offsets_deg"], np.rad2deg(expected), atol=1e-6)
