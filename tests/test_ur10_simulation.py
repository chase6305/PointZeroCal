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


def test_repeated_ur10_pose_cannot_calibrate_even_a_single_joint():
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))
    samples = np.tile([0.0, -1.2, 1.8, -1.4, -1.57, 0.0], (41, 1))
    result = JointZeroCalibrator(backend).calibrate(samples, fixed_joints=range(5))

    assert not result.converged
    assert result.rank == 0
    assert result.initial_point_rms == result.corrected_point_rms == 0


def test_dataset_generator_rejects_empty_orientation_set():
    backend = PinocchioKinematics(URDF, "ee_link")
    with pytest.raises(ValueError, match="N >= 1"):
        generate_fixed_point_dataset(backend, np.zeros(6), [])


@pytest.mark.parametrize("factory", [make_rpy_grid, make_rpy_samples])
@pytest.mark.parametrize("angle", [np.nan, np.inf, 0.0])
def test_orientation_generators_reject_invalid_angles(factory, angle):
    with pytest.raises(ValueError, match="max_angle_deg"):
        factory(max_angle_deg=angle)


@pytest.mark.parametrize("count", [3.5, True, 2])
def test_orientation_generators_reject_invalid_counts(count):
    with pytest.raises(ValueError, match="count"):
        make_rpy_samples(count=count)
    with pytest.raises(ValueError, match="levels"):
        make_rpy_grid(levels=count)


@pytest.mark.parametrize(
    "options", [{"max_iterations": 1.5}, {"tolerance": np.nan}, {"damping": np.inf}]
)
def test_dataset_generator_rejects_invalid_solver_options(options):
    backend = PinocchioKinematics(URDF, "ee_link")
    with pytest.raises(ValueError, match=next(iter(options))):
        generate_fixed_point_dataset(backend, np.zeros(6), [[0.0, 0.0, 0.0]], **options)


def test_dataset_accepts_convergence_on_the_last_allowed_ik_update():
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))
    seed = np.array([0.0, -1.2, 1.8, -1.4, -1.57, 0.0])
    nominal = backend.tcp_pose(seed)
    delta = np.array([1e-4, 0.0, 0.0])
    tolerance = 7.5e-5
    dataset = generate_fixed_point_dataset(
        backend,
        seed,
        [[value for value in delta]],
        max_iterations=1,
        tolerance=tolerance,
    )
    final = backend.tcp_pose(dataset.configurations_rad[0])
    target_rotation = backend._pin.rpy.rpyToMatrix(delta) @ nominal.rotation
    error = np.concatenate(
        [
            nominal.translation - final.translation,
            backend._pin.log3(target_rotation @ final.rotation.T),
        ]
    )

    assert np.linalg.norm(delta) > tolerance
    assert np.linalg.norm(error) <= tolerance
    assert dataset.position_errors[0] == pytest.approx(np.linalg.norm(error[:3]))


def test_dataset_accepts_nested_orientation_iterators():
    backend = PinocchioKinematics(URDF, "ee_link")
    seed = np.array([0.0, -1.2, 1.8, -1.4, -1.57, 0.0])
    deltas = ((value for value in [0.0, 0.0, 0.0]) for _ in range(2))
    dataset = generate_fixed_point_dataset(backend, seed, deltas)

    np.testing.assert_array_equal(dataset.configurations_rad, np.tile(seed, (2, 1)))


@pytest.mark.parametrize("deltas", [0.0, [0.0, 0.0, 0.0], [[0, 0], [0, 0, 0]]])
def test_dataset_reports_malformed_orientations_as_value_errors(deltas):
    backend = PinocchioKinematics(URDF, "ee_link")
    with pytest.raises(ValueError, match="rpy_deltas"):
        generate_fixed_point_dataset(backend, np.zeros(6), deltas)


def test_dataset_rejects_complex_seed_and_orientations():
    backend = PinocchioKinematics(URDF, "ee_link")
    with pytest.raises(ValueError, match="seed_configuration.*real"):
        generate_fixed_point_dataset(
            backend, np.ones(6, dtype=complex) * 1j, [[0, 0, 0]]
        )
    with pytest.raises(ValueError, match="rpy_deltas.*real"):
        generate_fixed_point_dataset(backend, np.zeros(6), np.array([[0, 0, 1j]]))


def test_ur10_noisy_fit_respects_measurement_tolerance():
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))
    dataset = generate_fixed_point_dataset(
        backend, [0.0, -1.2, 1.8, -1.4, -1.57, 0.0], make_rpy_samples()
    )
    expected = np.deg2rad([0.0, 1.20, -0.80, 1.50, -1.00, 2.00])
    measured = dataset.configurations_rad - expected
    measured += np.random.default_rng(42).normal(0, 1e-4, measured.shape)
    result = JointZeroCalibrator(backend, residual_tolerance=1e-3).calibrate(measured)

    assert result.converged
    assert result.rank == 5
    assert result.corrected_point_rms < result.initial_point_rms
    assert 1e-6 < result.corrected_point_rms < 1e-3
    np.testing.assert_allclose(result.offsets_rad, expected, atol=2e-3)


def test_sample_diagnostics_locate_a_bad_ur10_contact_and_preserve_row_order():
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))
    dataset = generate_fixed_point_dataset(
        backend, [0.0, -1.2, 1.8, -1.4, -1.57, 0.0], make_rpy_samples()
    )
    measured = dataset.configurations_rad - np.deg2rad([0, 1.2, -0.8, 1.5, -1, 2])
    bad_index = 17
    measured[bad_index, 2] += np.deg2rad(1)
    solver = JointZeroCalibrator(backend)
    result = solver.calibrate(measured)
    diagnostics = result.sample_diagnostics

    assert not result.converged
    assert diagnostics.worst_sample_index == bad_index
    assert diagnostics.max_corrected_point_error > 5 * result.corrected_point_rms
    np.testing.assert_allclose(
        np.sqrt(np.mean(diagnostics.corrected_point_errors**2)),
        result.corrected_point_rms,
        rtol=1e-12,
    )

    order = np.random.default_rng(42).permutation(len(measured))
    reordered = solver.calibrate(measured[order]).sample_diagnostics
    assert order[reordered.worst_sample_index] == bad_index
    np.testing.assert_allclose(
        reordered.corrected_point_errors,
        diagnostics.corrected_point_errors[order],
        atol=1e-10,
    )


def test_cli_prints_strict_json_diagnostics(tmp_path, capsys):
    backend = PinocchioKinematics(URDF, "ee_link", tcp_xyz=(0.08, 0.0, 0.16))
    dataset = dataset_for(backend)
    expected = np.deg2rad([0.0, 1.20, -0.80, 1.50, -1.00, 2.00])
    samples = tmp_path / "samples.csv"
    np.savetxt(
        samples, np.rad2deg(dataset.configurations_rad - expected), delimiter=","
    )

    exit_code = cli_main(
        [
            "--urdf",
            str(URDF),
            "--end-frame",
            "ee_link",
            "--tcp-xyz",
            "0.08",
            "0",
            "0.16",
            "--samples",
            str(samples),
            "--unit",
            "deg",
            "--fixed-joint",
            "Joint1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["message"] == "calibration converged"
    assert payload["sample_count"] == len(dataset.configurations_rad)
    assert payload["active_joint_names"][-1] == "Joint6"
    np.testing.assert_allclose(payload["offsets_deg"], np.rad2deg(expected), atol=1e-6)
