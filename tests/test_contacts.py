import json
from pathlib import Path

import numpy as np
import pytest

from point_zero_cal import (
    ContactGroup,
    JointZeroCalibrator,
    KnownDistance,
    PinocchioKinematics,
)
from point_zero_cal.simulation import generate_fixed_point_dataset, make_rpy_samples


@pytest.fixture
def arm(planar_urdf):
    return PinocchioKinematics(planar_urdf, "tip_link")


def contact_groups(offsets=(0.0, -0.025)):
    return [
        ContactGroup(
            "front/0",
            np.array([[0, np.pi / 2], [np.pi / 2, -np.pi / 2]]) - offsets,
            (3, 6),
        ),
        ContactGroup(
            "front/1",
            np.array([[-np.pi / 6, np.pi / 3], [np.pi / 6, -np.pi / 3]]) - offsets,
        ),
    ]


def known_distance():
    return KnownDistance("front/0", "front/1", float(np.hypot(np.sqrt(3) - 1, 1)))


def test_grouped_fit_recovers_shared_offsets_and_separate_centres(arm):
    result = JointZeroCalibrator(arm).calibrate_groups(
        contact_groups(), distances=[known_distance()]
    )

    assert result.converged
    assert result.rank == 1
    assert result.sample_count == 4
    np.testing.assert_allclose(result.offsets_rad, [0, -0.025], atol=1e-8)
    a, b = result.evaluation.groups
    np.testing.assert_allclose(a.contact_point_xyz_m, [1, 1, 0], atol=1e-8)
    np.testing.assert_allclose(b.contact_point_xyz_m, [np.sqrt(3), 0, 0], atol=1e-8)
    assert abs(result.evaluation.distances[0].error_m) < 1e-8
    assert result.evaluation.residual_rms_m < result.initial_evaluation.residual_rms_m
    assert result.sample_diagnostics is None
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    diagnostic = payload["evaluation"]["groups"][0]
    assert diagnostic["sample_csv_rows"] == [3, 6]
    assert (
        diagnostic["worst_sample_csv_row"] == [3, 6][diagnostic["worst_sample_index"]]
    )


@pytest.mark.parametrize("damping", [0.0, 1e-10, 0.1])
def test_single_group_uses_identical_solver_and_damping(arm, damping):
    group = contact_groups()[0]
    solver = JointZeroCalibrator(arm, damping=damping)
    single = solver.calibrate(group.samples)
    grouped = solver.calibrate_groups([group])

    np.testing.assert_array_equal(single.offsets_rad, grouped.offsets_rad)
    np.testing.assert_array_equal(single.singular_values, grouped.singular_values)
    assert single.iterations == grouped.iterations
    assert single.converged == grouped.converged
    assert single.corrected_point_rms == grouped.corrected_point_rms


def test_grouped_units_and_fixed_seed_are_preserved(arm):
    offsets = np.array([0.2, -0.025])
    groups = [
        ContactGroup(g.name, np.rad2deg(g.samples)) for g in contact_groups(offsets)
    ]
    seed = np.array([0.2, 0.0])
    result = JointZeroCalibrator(arm).calibrate_groups(
        groups,
        sample_unit="deg",
        fixed_joints=["joint1"],
        initial_offsets=seed,
        distances=[known_distance()],
    )
    assert result.converged
    np.testing.assert_allclose(result.offsets_rad, offsets, atol=1e-8)
    np.testing.assert_array_equal(result.initial_evaluation.offsets_rad, seed)
    np.testing.assert_array_equal(seed, [0.2, 0.0])


def repeated_groups(arm):
    true = np.array([[0.0, 0.4], [1.1, -0.5]])
    points = [arm.position_and_jacobian(q)[0].copy() for q in true]
    distance = float(np.linalg.norm(points[0] - points[1]))
    groups = [
        ContactGroup(str(i), np.tile(q - [0, 0.06], (3, 1))) for i, q in enumerate(true)
    ]
    return groups, KnownDistance("0", "1", distance)


def test_distance_adds_information_missing_from_within_group_residuals(arm):
    groups, distance = repeated_groups(arm)
    solver = JointZeroCalibrator(arm)
    without = solver.calibrate_groups(groups)
    with_distance = solver.calibrate_groups(groups, distances=[distance])

    assert without.rank == 0
    assert not without.converged
    assert with_distance.converged
    assert with_distance.rank == 1
    assert with_distance.corrected_point_rms == 0
    np.testing.assert_allclose(with_distance.offsets_rad, [0, 0.06], atol=1e-8)


def test_distance_does_not_make_common_base_rotation_observable(arm):
    groups, distance = repeated_groups(arm)
    result = JointZeroCalibrator(arm).calibrate_groups(
        groups,
        distances=[distance],
        fixed_joints=["joint2"],
        initial_offsets=[0, 0.06],
    )
    assert result.rank == 0
    assert not result.converged
    assert np.isinf(result.condition_number)


def test_all_joint_fit_still_reports_gauge_rank_deficiency(arm):
    result = JointZeroCalibrator(arm).calibrate_groups(
        contact_groups(),
        distances=[known_distance()],
        fixed_joints=[],
    )
    assert not result.converged
    assert result.rank < 2


def test_duplicate_samples_do_not_change_group_weight(arm):
    groups = contact_groups()
    noisy = groups[0].samples.copy()
    noisy[0, 0] += 0.01
    groups[0] = ContactGroup(groups[0].name, noisy)
    solver = JointZeroCalibrator(arm, damping=0.0, residual_tolerance=0.01)
    original = solver.calibrate_groups(groups, distances=[known_distance()])
    repeated = solver.calibrate_groups(
        [groups[0], ContactGroup(groups[1].name, np.tile(groups[1].samples, (25, 1)))],
        distances=[known_distance()],
    )
    np.testing.assert_allclose(repeated.offsets_rad, original.offsets_rad, atol=1e-9)
    assert repeated.residual_rms == pytest.approx(original.residual_rms, rel=1e-10)


def test_small_distance_weight_cannot_hide_wrong_measured_distance(arm):
    distance = known_distance()
    wrong = KnownDistance(
        distance.first, distance.second, distance.distance_m + 0.1, weight=1e-12
    )
    solver = JointZeroCalibrator(arm)
    result = solver.calibrate_groups(contact_groups(), distances=[wrong])
    assert result.corrected_point_rms < solver.residual_tolerance
    assert not result.converged
    assert result.message == "distance residual did not reach tolerance"
    assert abs(result.evaluation.distances[0].error_m) > 0.09
    relaxed = solver.calibrate_groups(
        contact_groups(), distances=[wrong], distance_tolerance=0.11
    )
    assert relaxed.converged


def test_group_and_distance_jacobians_match_finite_differences(arm):
    solver = JointZeroCalibrator(arm)
    q, plan = solver._prepare_groups(contact_groups(), [known_distance()], "rad")
    active = np.array([0, 1])
    state = solver._group_state(q, active, plan)
    for column in active:
        delta = np.zeros(2)
        delta[column] = 1e-6
        plus = solver._group_state(q + delta, active, plan)
        minus = solver._group_state(q - delta, active, plan)
        derivative = -(plus.error - minus.error) / 2e-6
        np.testing.assert_allclose(state.matrix[:, column], derivative, atol=1e-9)


def test_validation_uses_fixed_parameters_and_independent_contacts(arm, monkeypatch):
    solver = JointZeroCalibrator(arm)
    fit = solver.calibrate_groups(contact_groups(), distances=[known_distance()])
    held_out = ContactGroup(
        "left/0",
        np.array([[0, 2 * np.pi / 3], [2 * np.pi / 3, -2 * np.pi / 3]]) - [0, -0.025],
    )

    def forbid_optimization(*args, **kwargs):
        raise AssertionError("validation must not run optimization")

    monkeypatch.setattr(solver, "_solve_offsets", forbid_optimization)
    baseline = solver.evaluate_groups([held_out], offsets_rad=[0, 0])
    corrected = solver.evaluate_groups([held_out], offsets_rad=fit.offsets_rad)
    assert corrected.point_rms_m < 1e-8
    assert baseline.point_rms_m > 0.01
    perturbed = held_out.samples.copy()
    perturbed[0, 0] += 0.01
    bad = solver.evaluate_groups(
        [ContactGroup(held_out.name, perturbed)], offsets_rad=fit.offsets_rad
    )
    assert bad.point_rms_m > corrected.point_rms_m
    np.testing.assert_array_equal(bad.offsets_rad, fit.offsets_rad)
    assert not corrected.offsets_rad.flags.writeable
    assert not corrected.groups[0].point_errors_m.flags.writeable


def test_coincident_centres_can_be_evaluated_but_not_linearized(arm):
    groups = [ContactGroup(str(i), np.zeros((2, 2))) for i in range(2)]
    distance = KnownDistance("0", "1", 0.05)
    solver = JointZeroCalibrator(arm)
    report = solver.evaluate_groups(groups, offsets_rad=[0, 0], distances=[distance])
    assert report.distances[0].error_m == -0.05
    with pytest.raises(ValueError, match="coincident predicted centres"):
        solver.calibrate_groups(groups, distances=[distance])


@pytest.mark.parametrize(
    "groups, distances, kwargs, match",
    [
        ([], [], {}, "ContactGroup"),
        (["invalid"], [], {}, "ContactGroup"),
        ([contact_groups()[0]] * 2, [], {}, "unique"),
        (contact_groups(), [KnownDistance("missing", "front/0", 1)], {}, "unknown"),
        (contact_groups(), [known_distance(), known_distance()], {}, "duplicate"),
        (contact_groups(), [1], {}, "KnownDistance"),
        (contact_groups(), [], {"fixed_joints": [0, 1]}, "at least one"),
        (contact_groups(), [], {"sample_unit": "invalid"}, "sample_unit"),
        (contact_groups(), [], {"distance_tolerance": 0}, "distance_tolerance"),
        ([ContactGroup("bad", np.zeros((2, 3)))], [], {}, "shape"),
    ],
)
def test_invalid_group_problems_are_rejected(arm, groups, distances, kwargs, match):
    with pytest.raises(ValueError, match=match):
        JointZeroCalibrator(arm).calibrate_groups(groups, distances=distances, **kwargs)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True])
def test_known_distance_requires_positive_finite_length_and_weight(value):
    with pytest.raises(ValueError):
        KnownDistance("a", "b", value)
    with pytest.raises(ValueError):
        KnownDistance("a", "b", 0.05, weight=value)


def test_contact_observations_preserve_original_arrays_and_rows():
    samples = np.zeros((2, 2))
    rows = [3, 6]
    group = ContactGroup("a", samples, rows)
    samples[:] = 1
    rows[0] = 2
    assert group.csv_rows == (3, 6)
    np.testing.assert_array_equal(group.samples, 0)
    assert not group.samples.flags.writeable
    with pytest.raises(ValueError, match="csv_rows"):
        ContactGroup("a", samples, (1, 1))
    with pytest.raises(ValueError, match="finite"):
        ContactGroup("a", [[0, 0], [0, np.inf]])
    with pytest.raises(ValueError, match="different"):
        KnownDistance("a", "a", 0.05)


@pytest.mark.parametrize("noise_deg", [0.0, 0.002])
def test_ur10_two_socket_fit_generalizes_to_an_independent_placement(noise_deg):
    root = Path(__file__).resolve().parents[1]
    backend = PinocchioKinematics(
        root / "assets/UR10/UR10.urdf", "ee_link", tcp_xyz=(0.08, 0, 0.16)
    )
    seed = np.array([0, -1.2, 1.8, -1.4, -1.57, 0])
    origin = np.array(backend.tcp_pose(seed).translation)
    expected = np.deg2rad([0, 1.2, -0.8, 1.5, -1, 2])
    deltas = make_rpy_samples(count=21, max_angle_deg=40)
    rng = np.random.default_rng(123)
    groups = []
    for i, shift in enumerate(
        ([0, 0, 0], [0.05, 0, 0], [0, 0.08, 0.03], [0.05, 0.08, 0.03])
    ):
        dataset = generate_fixed_point_dataset(
            backend, seed, deltas, target_xyz=origin + shift
        )
        np.testing.assert_allclose(dataset.target_xyz, origin + shift, atol=1e-12)
        assert np.max(dataset.position_errors) < 1e-9
        measured = dataset.configurations_rad - expected
        measured += rng.normal(0, np.deg2rad(noise_deg), measured.shape)
        groups.append(ContactGroup(str(i), measured))
    solver = JointZeroCalibrator(backend, residual_tolerance=0.0001)
    fit = solver.calibrate_groups(groups[:2], distances=[KnownDistance("0", "1", 0.05)])
    assert fit.converged
    assert fit.rank == 5
    np.testing.assert_allclose(
        fit.offsets_deg, np.rad2deg(expected), atol=0.02 if noise_deg else 1e-6
    )
    held_out = groups[2:]
    nominal = solver.evaluate_groups(
        held_out, offsets_rad=np.zeros(6), distances=[KnownDistance("2", "3", 0.05)]
    )
    corrected = solver.evaluate_groups(
        held_out, offsets_rad=fit.offsets_rad, distances=[KnownDistance("2", "3", 0.05)]
    )
    assert corrected.point_rms_m < nominal.point_rms_m / 20
    assert corrected.point_rms_m < 0.0001
    assert abs(corrected.distances[0].error_m) < 0.0001
    assert all(g.point_rms_m < 0.0001 for g in corrected.groups)


def test_two_socket_distances_do_not_observe_an_axial_ur10_wrist():
    root = Path(__file__).resolve().parents[1]
    backend = PinocchioKinematics(
        root / "assets/UR10/UR10.urdf", "ee_link", tcp_xyz=(0, 0, 0.16)
    )
    seed = np.array([0, -1.2, 1.8, -1.4, -1.57, 0])
    origin = np.array(backend.tcp_pose(seed).translation)
    groups = []
    for i in range(2):
        dataset = generate_fixed_point_dataset(
            backend,
            seed,
            make_rpy_samples(count=21, max_angle_deg=40),
            target_xyz=origin + [0.05 * i, 0, 0],
        )
        groups.append(ContactGroup(str(i), dataset.configurations_rad))
    solver = JointZeroCalibrator(backend)
    result = solver.calibrate_groups(groups, distances=[KnownDistance("0", "1", 0.05)])
    assert result.rank == 4
    assert not result.converged
    fixed = solver.calibrate_groups(
        groups,
        distances=[KnownDistance("0", "1", 0.05)],
        fixed_joints=["Joint1", "Joint6"],
    )
    assert fixed.rank == 4
    assert fixed.converged


@pytest.mark.parametrize("target", [[1, 2], [1, 2, np.nan], [1j, 0, 0]])
def test_simulation_rejects_invalid_explicit_contact_targets(arm, target):
    with pytest.raises(ValueError, match="target_xyz"):
        generate_fixed_point_dataset(arm, [0, 0], [[0, 0, 0]], target_xyz=target)
