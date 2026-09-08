"""Geometric origin calibration against synthetic perturbed-URDF data."""

import json

import numpy as np
import pytest

from point_zero_cal import (
    ContactGroup,
    KnownDistance,
    PinocchioKinematics,
)
from point_zero_cal.geometric import (
    GeometricCalibrator,
    GeometricParameter,
)
from point_zero_cal.geometric_backend import PinocchioGeometricKinematics

PLANAR = """<?xml version="1.0"?>
<robot name="planar">
  <link name="base"/>
  <link name="link1"/>
  <link name="tool"/>
  <link name="tip_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="tool"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="tip" type="fixed">
    <parent link="tool"/><child link="tip_link"/><origin xyz="1 0 0"/>
  </joint>
</robot>
"""

TRUTH = (0.005, 0.004)
POINTS = (np.array([1.0, 1.0, 0.0]), np.array([np.sqrt(3), 0.0, 0.0]))


def write_planar_urdfs(tmp_path, perturb_xyz="1.005 0.004 0"):
    nominal = tmp_path / "nominal.urdf"
    true = tmp_path / "true.urdf"
    nominal.write_text(PLANAR, encoding="utf-8")
    true.write_text(
        PLANAR.replace(
            '<origin xyz="1 0 0"/><axis xyz="0 0 1"/>',
            f'<origin xyz="{perturb_xyz}"/><axis xyz="0 0 1"/>',
        ),
        encoding="utf-8",
    )
    return nominal, true


def ik_touch(backend, point, seed):
    """Resolved-rate IK of the planar arm onto ``point``."""
    q = np.asarray(seed, dtype=float)
    for _ in range(200):
        position, jacobian = backend.position_and_jacobian(q)
        error = point - position
        if np.linalg.norm(error) < 1e-12:
            return q
        q += np.linalg.solve(jacobian[:2, :2], error[:2])
    raise RuntimeError("planar IK did not converge")


def planar_groups(true_backend):
    groups = []
    for index, point in enumerate(POINTS):
        configurations = np.array(
            [ik_touch(true_backend, point, seed) for seed in ([0.2, 1.5], [1.2, -1.2])]
        )
        groups.append(ContactGroup(str(index), configurations))
    return groups


def planar_distance():
    return KnownDistance("0", "1", float(np.linalg.norm(POINTS[0] - POINTS[1])))


@pytest.fixture
def planar_setup(tmp_path):
    nominal_path, true_path = write_planar_urdfs(tmp_path)
    true_backend = PinocchioKinematics(true_path, "tip_link")
    return nominal_path, true_backend, planar_groups(true_backend)


def make_solver(nominal_path, *, sigma=100.0, **parameter_kwargs):
    parameters = [
        GeometricParameter("joint2", "x", sigma, **parameter_kwargs),
        GeometricParameter("joint2", "y", sigma, **parameter_kwargs),
    ]
    backend = PinocchioGeometricKinematics(nominal_path, "tip_link", parameters)
    return GeometricCalibrator(backend, residual_tolerance=1e-6)


def test_loose_priors_recover_injected_origin_deviations(planar_setup):
    nominal_path, _, groups = planar_setup
    fit = make_solver(nominal_path).calibrate_groups(
        groups, distances=[planar_distance()]
    )
    assert fit.converged
    assert fit.data_rank == 2
    assert fit.message == "geometric calibration converged"
    np.testing.assert_allclose([p.estimate for p in fit.parameters], TRUTH, rtol=1e-2)
    assert fit.evaluation.residual_rms_m < 1e-5
    assert fit.nullspace_directions == ()
    for parameter in fit.parameters:
        assert parameter.unit == "m"
        assert parameter.active


def test_tight_priors_hold_parameters_near_nominal(planar_setup):
    nominal_path, _, groups = planar_setup
    fit = make_solver(nominal_path, sigma=0.001).calibrate_groups(
        groups, distances=[planar_distance()]
    )
    assert not fit.converged
    assert fit.data_rank == 2  # priors never fake observability
    assert fit.message == "contact group residual did not reach tolerance"
    assert max(abs(p.estimate) for p in fit.parameters) < 1e-6
    assert fit.prior_penalty < 1e-5


def test_parameter_bounds_clamp_the_accepted_estimate(planar_setup):
    nominal_path, _, groups = planar_setup
    parameters = [
        GeometricParameter("joint2", "x", 100.0, upper=0.001),
        GeometricParameter("joint2", "y", 100.0),
    ]
    backend = PinocchioGeometricKinematics(nominal_path, "tip_link", parameters)
    fit = GeometricCalibrator(backend, residual_tolerance=0.01).calibrate_groups(
        groups, distances=[planar_distance()]
    )
    assert not fit.converged
    assert fit.parameters[0].estimate == 0.001
    np.testing.assert_allclose(fit.parameters[1].estimate, TRUTH[1], rtol=1e-2)


def test_state_rows_match_finite_differences(planar_setup):
    nominal_path, _, groups = planar_setup
    solver = make_solver(nominal_path)
    q, plan = solver._prepare_groups(groups, [planar_distance()], "rad")
    active = solver._active_indices()
    state = solver._state(np.zeros(len(active)), q, plan, active)
    for column in range(len(active)):
        delta = np.zeros(len(active))
        delta[column] = 1e-7
        plus = solver._state(delta, q, plan, active)
        minus = solver._state(-delta, q, plan, active)
        derivative = -(plus.error - minus.error) / 2e-7
        significant = state.matrix[:, column] != 0
        np.testing.assert_allclose(
            state.matrix[:, column][significant],
            derivative[significant],
            atol=1e-8,
        )


def test_evaluate_groups_fixes_parameters_without_fitting(planar_setup, monkeypatch):
    nominal_path, _, groups = planar_setup
    solver = make_solver(nominal_path)

    def forbid_optimization(*args, **kwargs):
        raise AssertionError("validation must not run optimization")

    monkeypatch.setattr(
        "point_zero_cal.geometric.backtracking_solve", forbid_optimization
    )
    nominal_report = solver.evaluate_groups(
        groups, parameters=[0.0, 0.0], distances=[planar_distance()]
    )
    corrected_report = solver.evaluate_groups(
        groups, parameters=list(TRUTH), distances=[planar_distance()]
    )
    assert corrected_report.point_rms_m < 1e-9
    assert nominal_report.point_rms_m > 1e-3
    np.testing.assert_array_equal(corrected_report.offsets_rad, [0.0, 0.0])
    assert not corrected_report.offsets_rad.flags.writeable


def test_group_sigma_weighs_the_geometric_objective(planar_setup):
    nominal_path, _, groups = planar_setup
    noisy = groups[1].samples.copy()
    noisy[0, 0] += 0.01
    noisy[1, 1] -= 0.008
    parameters = [
        GeometricParameter("joint2", "x", 100.0),
        GeometricParameter("joint2", "y", 100.0),
    ]
    backend = PinocchioGeometricKinematics(nominal_path, "tip_link", parameters)
    solver = GeometricCalibrator(backend, residual_tolerance=0.02)
    equal = solver.calibrate_groups(
        [groups[0], ContactGroup("1", noisy)], distances=[planar_distance()]
    )
    weighted = solver.calibrate_groups(
        [
            ContactGroup("0", groups[0].samples, sigma_m=0.0005),
            ContactGroup("1", noisy, sigma_m=0.02),
        ],
        distances=[planar_distance()],
    )
    assert equal.converged and weighted.converged
    truth = np.array(TRUTH)
    equal_error = np.linalg.norm([p.estimate for p in equal.parameters] - truth)
    weighted_error = np.linalg.norm([p.estimate for p in weighted.parameters] - truth)
    assert weighted_error < equal_error / 50
    assert weighted.evaluation.groups[1].sigma_m == 0.02


def test_result_round_trips_as_strict_json(planar_setup):
    nominal_path, _, groups = planar_setup
    fit = make_solver(nominal_path).calibrate_groups(
        groups, distances=[planar_distance()]
    )
    payload = json.loads(json.dumps(fit.to_dict(), allow_nan=False))
    assert payload["converged"] is True
    assert payload["data_rank"] == 2
    assert [p["joint_name"] for p in payload["parameters"]] == [
        "joint2",
        "joint2",
    ]
    assert payload["parameters"][0]["unit"] == "m"
    assert "evaluation" in payload and "initial_evaluation" in payload


def test_geometric_parameter_validation():
    with pytest.raises(ValueError, match="component"):
        GeometricParameter("j", "vx", 0.1)
    with pytest.raises(ValueError, match="prior_scale"):
        GeometricParameter("j", "x", 0)
    with pytest.raises(ValueError, match="finite"):
        GeometricParameter("j", "x", 0.1, nominal=np.nan)
    with pytest.raises(ValueError, match="lower"):
        GeometricParameter("j", "x", 0.1, lower=1, upper=0)
    with pytest.raises(ValueError, match="boolean"):
        GeometricParameter("j", "x", 0.1, active=1)
    parameter = GeometricParameter("j", "roll", 0.1, nominal=-0.02)
    assert parameter.unit == "rad"
    assert parameter.label == "j.roll"


class _StubGeometricBackend:
    """Minimal backend for calibrator-level parameter validation."""

    dof = 2
    joint_names = ("joint1", "joint2")

    def __init__(self, parameters):
        self.geometric_parameters = tuple(parameters)


def test_calibrator_rejects_empty_or_duplicate_parameter_sets():
    with pytest.raises(ValueError, match="at least one"):
        GeometricCalibrator(_StubGeometricBackend([]))
    with pytest.raises(ValueError, match="unique"):
        GeometricCalibrator(
            _StubGeometricBackend([GeometricParameter("j", "x", 0.1)] * 2)
        )


def test_no_active_parameters_are_rejected(planar_setup):
    nominal_path, _, groups = planar_setup
    parameters = [GeometricParameter("joint2", "x", 0.1, active=False)]
    backend = PinocchioGeometricKinematics(nominal_path, "tip_link", parameters)
    solver = GeometricCalibrator(backend)
    with pytest.raises(ValueError, match="at least one parameter"):
        solver.calibrate_groups(groups)


def test_invalid_seeds_and_distances_are_rejected(planar_setup):
    nominal_path, _, groups = planar_setup
    solver = make_solver(nominal_path)
    with pytest.raises(ValueError, match="initial_parameters"):
        solver.calibrate_groups(groups, initial_parameters=[0.0])
    with pytest.raises(ValueError, match="unknown"):
        solver.calibrate_groups(groups, distances=[KnownDistance("0", "missing", 0.5)])
    with pytest.raises(ValueError, match="sample_unit"):
        solver.calibrate_groups(groups, sample_unit="gon")
    with pytest.raises(ValueError, match="parameters must have shape"):
        solver.evaluate_groups(groups, parameters=[0.0, 0.0, 0.0])


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize("seed", [-0.02, 0.02])
def test_out_of_bounds_seeds_are_rejected(planar_setup, active, seed):
    nominal_path, _, groups = planar_setup
    backend = PinocchioGeometricKinematics(
        nominal_path, "tip_link",
        [
            GeometricParameter("joint2", "x", 1, active=active, lower=-0.01, upper=0.01),
            GeometricParameter("joint2", "y", 1),
        ],
    )
    with pytest.raises(ValueError, match="initial_parameters.*joint2.x.*bounds"):
        GeometricCalibrator(backend).calibrate_groups(groups, initial_parameters=[seed, 0])


def test_ur10_gauge_rank_deficiency_reports_nullspace_directions():
    """Fixed-point data cannot separate all TCP mounting rotations."""
    from pathlib import Path

    from point_zero_cal.simulation import (
        generate_fixed_point_dataset,
        make_rpy_samples,
    )

    root = Path(__file__).resolve().parents[1]
    urdf = root / "assets/UR10/UR10.urdf"
    tcp = (0.08, 0, 0.16)
    backend = PinocchioKinematics(urdf, "ee_link", tcp_xyz=tcp)
    seed = np.array([0, -1.2, 1.8, -1.4, -1.57, 0])
    origin = np.array(backend.tcp_pose(seed).translation)
    deltas = make_rpy_samples(count=21, max_angle_deg=40)
    groups = []
    for index, shift in enumerate(([0, 0, 0], [0.05, 0, 0])):
        dataset = generate_fixed_point_dataset(
            backend, seed, deltas, target_xyz=origin + shift
        )
        groups.append(ContactGroup(str(index), dataset.configurations_rad))
    parameters = [
        GeometricParameter("jointEE", "roll", 0.5),
        GeometricParameter("jointEE", "pitch", 0.5),
        GeometricParameter("jointEE", "yaw", 0.5),
    ]
    geometric = PinocchioGeometricKinematics(urdf, "ee_link", parameters, tcp_xyz=tcp)
    fit = GeometricCalibrator(geometric, residual_tolerance=1e-6).calibrate_groups(
        groups, distances=[KnownDistance("0", "1", 0.05)]
    )
    assert not fit.converged
    assert fit.data_rank == 2
    assert (
        fit.message == "insufficient data observability: rank 2 < 3 active parameters"
    )
    assert fit.nullspace_directions
    assert any("jointEE." in direction for direction in fit.nullspace_directions)
    assert np.isinf(fit.data_condition_number)


def test_reports_reuse_solver_evaluations(planar_setup, monkeypatch):
    nominal_path, _, groups = planar_setup
    solver = make_solver(nominal_path)
    evaluate = solver._evaluate
    state = solver._state
    counts = {"evaluate": 0, "state": 0}

    def counted_evaluate(*args):
        counts["evaluate"] += 1
        return evaluate(*args)

    def counted_state(*args):
        counts["state"] += 1
        return state(*args)

    monkeypatch.setattr(solver, "_evaluate", counted_evaluate)
    monkeypatch.setattr(solver, "_state", counted_state)
    fit = solver.calibrate_groups(groups, distances=[planar_distance()])
    assert fit.converged
    assert counts["evaluate"] == counts["state"]
    for parameters, cached in (
        ([0.0, 0.0], fit.initial_evaluation),
        ([p.estimate for p in fit.parameters], fit.evaluation),
    ):
        report = solver.evaluate_groups(
            groups, parameters, distances=[planar_distance()]
        )
        assert report.to_dict() == cached.to_dict()


def test_wide_data_matrix_reports_complete_nullspace():
    class Backend:
        dof = 1
        joint_names = ("joint",)
        geometric_parameters = tuple(
            GeometricParameter(f"joint{k}", "x", 1.0) for k in range(8)
        )

        def position_and_geometric_jacobian(self, configuration, parameters):
            # Two observations provide six rows for eight parameters, with
            # just one observable combination after group centering.
            jacobian = np.zeros((3, 8))
            jacobian[0, 0] = configuration[0]
            return jacobian @ parameters, jacobian

    fit = GeometricCalibrator(Backend()).calibrate_groups(
        [ContactGroup("contact", [[-1.0], [1.0]])]
    )
    assert not fit.converged
    assert fit.data_rank == 1
    assert len(fit.nullspace_directions) == 7
    assert np.isinf(fit.data_condition_number)
