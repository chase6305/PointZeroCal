"""Regression tests for shared contact reductions and weighted rows."""

import numpy as np
import pytest

from point_zero_cal import ContactGroup, KnownDistance, _groups


def test_opposite_extreme_positions_have_representable_residuals():
    positions = np.array([[-1e308, 0, 0], [1e308, 0, 0]])
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(_groups.center(positions), positions)
        assert _groups.point_rms(positions) == 1e308
        np.testing.assert_array_equal(_groups.point_errors(positions), [1e308, 1e308])


def test_unrepresentable_centered_residuals_still_raise():
    positions = np.array([[-1.7e308, 0, 0], [1.7e308, 0, 0], [1.7e308, 0, 0]])
    with pytest.raises(ValueError, match="floating-point range"):
        _groups.center(positions)


def test_large_distance_weight_keeps_representable_rows():
    _, plan = _groups.prepare_groups(
        [ContactGroup("a", [[0], [0]]), ContactGroup("b", [[1], [1]])],
        [KnownDistance("a", "b", 0.5, weight=1e308)],
        "rad",
        1,
    )
    positions = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    jacobians = np.zeros((4, 3, 1))
    jacobians[2:, 0, 0] = 1
    with np.errstate(over="raise", invalid="raise"):
        matrix, residual = _groups.group_state_system(
            positions, jacobians, plan, np.array([0])
        )
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(residual))
    np.testing.assert_allclose(matrix[-1] / 1e154, [2])
    np.testing.assert_allclose(residual[-1] / 1e154, 1)


def test_group_report_reuses_centred_positions(monkeypatch):
    _, plan = _groups.prepare_groups(
        [ContactGroup("a", [[0], [1], [2]]), ContactGroup("b", [[0], [1]])],
        [],
        "rad",
        1,
    )
    positions = 1e10 + np.arange(15, dtype=float).reshape(5, 3) / 10
    expected = [
        (_groups.point_rms(positions[s]), _groups.point_errors(positions[s]))
        for s in plan.sections
    ]
    center = _groups.center
    calls = []

    def counted_center(values):
        calls.append(len(values))
        return center(values)

    monkeypatch.setattr(_groups, "center", counted_center)
    report = _groups.group_report(positions, plan, np.zeros(1))
    assert calls == [3, 2]
    for group, (rms, errors) in zip(report.groups, expected, strict=True):
        assert group.point_rms_m == rms
        np.testing.assert_array_equal(group.point_errors_m, errors)


@pytest.mark.parametrize(
    "indices", [[0, 1, 2], [1, 2], [0, 2], [2, 0], [1], [0, 0, 2]]
)
def test_group_system_keeps_active_column_order_and_detaches_outputs(
    indices, monkeypatch
):
    rng = np.random.default_rng(204)
    groups = [ContactGroup(str(i), np.zeros((7, 3)), sigma_m=0.1 + i) for i in range(3)]
    distances = [KnownDistance(str(i), str(j), 1) for i in range(3) for j in range(i)]
    _, plan = _groups.prepare_groups(groups, distances, "rad", 3)
    positions, jacobians = rng.normal(size=(21, 3)), rng.normal(size=(21, 3, 3))
    saved_positions, saved_jacobians = positions.copy(), jacobians.copy()
    active = np.array(indices)
    actual = _groups.group_state_system(positions, jacobians, plan, active)
    with monkeypatch.context() as context:
        context.setattr(_groups, "active_columns", lambda active, width: active)
        expected = _groups.group_state_system(positions, jacobians, plan, active)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(positions, saved_positions)
    np.testing.assert_array_equal(jacobians, saved_jacobians)
    assert not np.shares_memory(actual[0], jacobians)
    assert not np.shares_memory(actual[1], positions)


@pytest.mark.parametrize("weighted", [False, True])
def test_independent_extreme_groups_do_not_require_representable_separation(weighted):
    groups = [
        ContactGroup("a", [[0], [1]], sigma_m=0.5 if weighted else None),
        ContactGroup("b", [[0], [1]], sigma_m=2.0 if weighted else None),
    ]
    _, plan = _groups.prepare_groups(groups, (), "rad", 1)
    positions = np.array(
        [
            [-1e308, -1, 0],
            [-1e308, 1, 0],
            [1e308, -2, 0],
            [1e308, 2, 0],
        ]
    )
    jacobians = np.zeros((4, 3, 1))
    jacobians[:, 1, 0] = [-1, 1, -2, 2]
    active = np.array([0])
    with np.errstate(over="raise", invalid="raise"):
        matrix, residual = _groups.group_state_system(
            positions, jacobians, plan, active
        )
        information = _groups.contact_information_matrix(
            [positions[s] for s in plan.sections],
            [jacobians[s] for s in plan.sections],
            plan,
            active,
        )
        report = _groups.group_report(positions, plan, np.zeros(1))
    expected = np.array([0, -1, 0, 0, 1, 0, 0, -2, 0, 0, 2, 0], dtype=float)
    if weighted:
        expected[:6] /= 0.5
        expected[6:] /= 2.0
    np.testing.assert_array_equal(residual, expected)
    np.testing.assert_array_equal(matrix[:, 0], expected)
    np.testing.assert_array_equal(information, matrix)
    assert report.point_rms_m == pytest.approx(np.sqrt(2.5))
    np.testing.assert_array_equal(report.groups[0].contact_point_xyz_m, [-1e308, 0, 0])
    np.testing.assert_array_equal(report.groups[1].contact_point_xyz_m, [1e308, 0, 0])


def test_unrepresentable_separation_is_still_rejected_when_distance_is_requested():
    _, plan = _groups.prepare_groups(
        [ContactGroup("a", [[0], [1]]), ContactGroup("b", [[0], [1]])],
        [KnownDistance("a", "b", 1)],
        "rad",
        1,
    )
    positions = np.array([[-1e308, 0, 0]] * 2 + [[1e308, 0, 0]] * 2)
    with pytest.raises(ValueError, match="contact centres"):
        _groups.group_state_system(positions, np.zeros((4, 3, 1)), plan, np.array([0]))


def test_information_without_distances_does_not_need_position_residuals():
    _, plan = _groups.prepare_groups([ContactGroup("a", [[0], [1], [2]])], (), "rad", 1)
    positions = np.array([[-1.7e308, 0, 0], [1.7e308, 0, 0], [1.7e308, 0, 0]])
    jacobians = np.zeros((3, 3, 1))
    jacobians[:, 0, 0] = [-1, 0, 1]
    active = np.array([0])
    with np.errstate(over="raise", invalid="raise"):
        information = _groups.contact_information_matrix(
            [positions],
            [jacobians],
            plan,
            active,
        )
    np.testing.assert_array_equal(information, jacobians.reshape(-1, 1))
    with pytest.raises(ValueError, match="centered kinematics"):
        _groups.group_state_system(positions, jacobians, plan, active)


@pytest.mark.parametrize("far_coordinate", [-1e16, -1e308, 1e308])
@pytest.mark.parametrize("reverse", [False, True])
def test_distance_spacing_is_independent_of_unrelated_remote_group(
    far_coordinate, reverse
):
    endpoints = ("b", "a") if reverse else ("a", "b")
    _, plan = _groups.prepare_groups(
        [ContactGroup(name, [[0], [0]]) for name in ("far", "a", "b")],
        [KnownDistance(*endpoints, 2)],
        "rad",
        1,
    )
    positions = np.array(
        [[far_coordinate, 0, 0]] * 2 + [[1e16, 0, 0]] * 2 + [[1e16 + 2, 0, 0]] * 2
    )
    jacobians = np.zeros((6, 3, 1))
    jacobians[4:, 0, 0] = 1
    with np.errstate(over="raise", invalid="raise"):
        matrix, residual = _groups.group_state_system(
            positions, jacobians, plan, np.array([0])
        )
        report = _groups.group_report(positions, plan, np.zeros(1))
        information = _groups.contact_information_matrix(
            [positions[s] for s in plan.sections],
            [jacobians[s] for s in plan.sections],
            plan,
            np.array([0]),
        )
    assert report.distances[0].predicted_distance_m == 2
    assert report.distances[0].error_m == 0
    np.testing.assert_array_equal(residual, 0)
    assert matrix[-1, 0] == pytest.approx(np.sqrt(6))
    np.testing.assert_array_equal(information, matrix)


def test_distance_preserves_spacing_between_rounded_world_centroids():
    _, plan = _groups.prepare_groups(
        [
            ContactGroup("far", [[0], [0]]),
            ContactGroup("a", [[0], [0]]),
            ContactGroup("b", [[0], [0], [0]]),
        ],
        [KnownDistance("a", "b", 1 / 3)],
        "rad",
        1,
    )
    positions = np.array(
        [
            [-1e16, 0, 0],
            [-1e16, 0, 0],
            [1e16, 0, 0],
            [1e16 + 2, 0, 0],
            [1e16, 0, 0],
            [1e16, 0, 0],
            [1e16 + 2, 0, 0],
        ]
    )
    report = _groups.group_report(positions, plan, np.zeros(1))
    assert (
        report.groups[1].contact_point_xyz_m[0]
        == report.groups[2].contact_point_xyz_m[0]
    )
    assert report.distances[0].predicted_distance_m == pytest.approx(1 / 3)
    assert abs(report.distances[0].error_m) < 1e-15


def test_distance_displacement_handles_overflowing_representative_difference():
    _, plan = _groups.prepare_groups(
        [
            ContactGroup("a", [[0], [0]], sigma_m=1e308),
            ContactGroup("b", [[0], [0]], sigma_m=1e308),
        ],
        [KnownDistance("a", "b", 1)],
        "rad",
        1,
    )
    positions = np.array([[-1e308, 0, 0], [1e308, 0, 0], [1e308, 1, 0], [-1e308, 1, 0]])
    with np.errstate(over="raise", invalid="raise"):
        matrix, residual = _groups.group_state_system(
            positions, np.zeros((4, 3, 1)), plan, np.array([0])
        )
        report = _groups.group_report(positions, plan, np.zeros(1))
    assert np.all(np.isfinite(matrix)) and np.all(np.isfinite(residual))
    assert report.distances[0].predicted_distance_m == 1
    assert report.distances[0].error_m == 0


def test_tiny_group_sigma_keeps_finite_weighted_rows():
    sigma = np.nextafter(0.0, 1.0)
    _, plan = _groups.prepare_groups(
        [ContactGroup("a", [[0], [1]], sigma_m=sigma)], (), "rad", 1
    )
    positions = np.array([[0, 0, 0], [2 * sigma, 0, 0]])
    jacobians = positions[:, :, None].copy()
    with np.errstate(over="raise", invalid="raise"):
        matrix, residual = _groups.group_state_system(
            positions, jacobians, plan, np.array([0])
        )
    expected = np.array([-1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(matrix[:, 0], expected)
    np.testing.assert_array_equal(residual, expected)


def test_tiny_distance_sigma_keeps_finite_weighted_rows():
    sigma = np.nextafter(0.0, 1.0)
    _, plan = _groups.prepare_groups(
        [ContactGroup("a", [[0], [0]]), ContactGroup("b", [[1], [1]])],
        [KnownDistance("a", "b", sigma, sigma_m=sigma)],
        "rad",
        1,
    )
    positions = np.array([[0, 0, 0]] * 2 + [[2 * sigma, 0, 0]] * 2)
    jacobians = np.zeros((4, 3, 1))
    jacobians[2:, 0, 0] = sigma
    with np.errstate(over="raise", invalid="raise"):
        matrix, residual = _groups.group_state_system(
            positions, jacobians, plan, np.array([0])
        )
    assert matrix[-1, 0] == residual[-1] == 2
    np.testing.assert_array_equal(matrix[:-1], 0)
    np.testing.assert_array_equal(residual[:-1], 0)


def test_large_mean_jacobians_do_not_overflow_roundoff_threshold():
    _, plan = _groups.prepare_groups(
        [ContactGroup("a", [[0], [0]]), ContactGroup("b", [[1], [1]])],
        [KnownDistance("a", "b", 0.5)],
        "rad",
        1,
    )
    positions = np.array([[0, 0, 0]] * 2 + [[1, 0, 0]] * 2, dtype=float)
    jacobians = np.zeros((4, 3, 1))
    jacobians[:2, 0, 0] = 1e308
    jacobians[2:, 0, 0] = 9e307
    with np.errstate(over="raise", invalid="raise"):
        matrix, residual = _groups.group_state_system(
            positions, jacobians, plan, np.array([0])
        )
    assert matrix[-1, 0] == pytest.approx(-2e307)
    assert residual[-1] == 1


@pytest.mark.parametrize("sigma", [np.nextafter(0.0, 1.0), 1e308])
def test_extreme_noise_scaling_preserves_zero_rows(sigma):
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(
            _groups.weighted_values(np.zeros((3, 2)), 0.5, sigma), 0
        )


def test_unreferenced_positions_need_not_be_centered_for_distance_information():
    _, plan = _groups.prepare_groups(
        [
            ContactGroup("far", [[0], [0], [0]]),
            ContactGroup("a", [[0], [0]]),
            ContactGroup("b", [[0], [0]]),
        ],
        [KnownDistance("a", "b", 1)],
        "rad",
        1,
    )
    positions = [
        np.array([[-1.7e308, 0, 0], [1.7e308, 0, 0], [1.7e308, 0, 0]]),
        np.zeros((2, 3)),
        np.array([[1, 0, 0]] * 2, dtype=float),
    ]
    jacobians = [np.zeros((len(p), 3, 1)) for p in positions]
    jacobians[0][:, 0, 0] = [-1, 0, 1]
    jacobians[2][:, 0, 0] = 1
    active = np.array([0])
    with np.errstate(over="raise", invalid="raise"):
        matrix = _groups.contact_information_matrix(positions, jacobians, plan, active)
        information = _groups.candidate_information(
            matrix, jacobians, plan, active, 0, group_positions=positions
        )
        candidate_jacobian = np.array([[2], [0], [0]])
        candidate_position = np.array([1.7e308, 0, 0])
        compressed = information.append(candidate_jacobian, candidate_position)
        full_positions = [np.vstack((positions[0], candidate_position)), *positions[1:]]
        full_jacobians = [
            np.concatenate((jacobians[0], candidate_jacobian[None])),
            *jacobians[1:],
        ]
        full = _groups.contact_information_matrix(
            full_positions, full_jacobians, plan, active
        )
    assert compressed is not None
    np.testing.assert_allclose(
        np.linalg.svd(compressed, compute_uv=False),
        np.linalg.svd(full, compute_uv=False),
        rtol=1e-14,
    )


@pytest.mark.parametrize("sigma", [4.0, 1e308])
def test_distance_projection_can_overflow_before_finite_weighted_rows(sigma):
    first = np.array([[1e308, 1e-100], [1e308, 0], [0, 0]])
    second = np.array([[-1e308, 0], [-1e308, 0], [0, 0]])
    distance = KnownDistance("a", "b", 1, sigma_m=sigma)
    with np.errstate(over="raise", invalid="raise"):
        row, length = _groups.distance_row(
            np.array([1, 1, 0]), first, second, distance, 4
        )
    expected = np.array(
        [
            np.sqrt(np.longdouble(2)) * np.longdouble(1e308) * 4,
            np.longdouble(1e-100) * np.sqrt(np.longdouble(2)),
        ]
    ) / np.longdouble(sigma)
    np.testing.assert_allclose(row, expected.astype(float), rtol=1e-14, atol=0)
    assert length == pytest.approx(np.sqrt(2))


def test_unrepresentable_distance_length_is_rejected_without_warnings():
    delta = np.array([1.5e308, 1.5e308, 0])
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="contact distance exceeds"):
            _groups.distance_length(delta)


@pytest.mark.parametrize("world", [0.0, 1e16, 1e300])
def test_distance_candidate_preserves_small_centroid_separations(world):
    unit = max(1.0, np.spacing(world))
    origin = np.full(3, world)
    positions = [
        origin + np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]]) * unit,
        origin + np.array([[4, 2, 0], [6, 2, 0], [8, 2, 0]]) * unit,
        np.full((3, 3), 1e308),
    ]
    rng = np.random.default_rng(121)
    jacobians = [rng.normal(size=(3, 3, 4)) for _ in positions]
    groups = [
        ContactGroup(str(index), np.zeros((3, 4)), sigma_m=sigma)
        for index, sigma in enumerate((0.005, 0.02, None))
    ]
    distance = KnownDistance("0", "1", np.sqrt(20) * unit, weight=0.7, sigma_m=0.03)
    _, plan = _groups.prepare_groups(groups, [distance], "rad", 4)
    active = np.arange(4)
    matrix = _groups.contact_information_matrix(positions, jacobians, plan, active)
    information = _groups.candidate_information(
        matrix, jacobians, plan, active, 0, group_positions=positions
    )
    candidate_position = origin + np.array([6, 0, 0]) * unit
    candidate_jacobian = rng.normal(size=(3, 4))
    compressed = information.append(candidate_jacobian, candidate_position)
    augmented_positions = [
        np.vstack((positions[0], candidate_position)),
        *positions[1:],
    ]
    augmented_jacobians = [
        np.concatenate((jacobians[0], candidate_jacobian[None])),
        *jacobians[1:],
    ]
    full = _groups.contact_information_matrix(
        augmented_positions, augmented_jacobians, plan, active
    )
    direction = np.array([-3, -2, 0]) / np.sqrt(13)
    expected_distance_row = (
        direction
        @ (augmented_jacobians[0].mean(axis=0) - jacobians[1].mean(axis=0))
        * np.sqrt(10)
        * np.sqrt(0.7)
        / 0.03
    )
    np.testing.assert_allclose(compressed[-1], expected_distance_row, rtol=1e-14)
    np.testing.assert_allclose(full[-1], expected_distance_row, rtol=1e-14)
    np.testing.assert_allclose(
        np.linalg.svd(compressed, compute_uv=False),
        np.linalg.svd(full, compute_uv=False),
        rtol=1e-14,
    )


@pytest.mark.parametrize("count", [1, 2, 3, 7, 28])
def test_distance_rows_match_scalar_projection_with_mixed_noise(count):
    rng = np.random.default_rng(123)
    deltas = rng.normal(size=(count, 3))
    first, second = rng.normal(size=(2, count, 3, 5))
    distances = [
        KnownDistance(
            str(index),
            str(index + 1),
            1,
            weight=index + 0.5,
            sigma_m=None if index % 2 else 0.005 * (index + 1),
        )
        for index in range(count)
    ]
    expected = [
        _groups.distance_row(delta, a, b, distance, 40)
        for delta, a, b, distance in zip(deltas, first, second, distances, strict=True)
    ]
    rows, lengths = _groups.distance_rows(deltas, first, second, distances, 40)
    np.testing.assert_allclose(
        rows, [item[0] for item in expected], rtol=1e-13, atol=1e-12
    )
    np.testing.assert_array_equal(lengths, [item[1] for item in expected])


def test_distance_rows_preserve_extreme_scaling_and_tiny_columns():
    deltas = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=float)
    first, second = np.zeros((4, 3, 2)), np.zeros((4, 3, 2))
    first[0] = [[1e308, 1e-100], [1e308, 0], [0, 0]]
    second[0] = [[-1e308, 0], [-1e308, 0], [0, 0]]
    first[1, 0] = [3e-308, 1e-323]
    first[2, 0] = [1e308, 1e100]
    first[3, 0, 0] = 1e308
    distances = [
        KnownDistance("a", "b", 1, sigma_m=4),
        KnownDistance("b", "c", 1, sigma_m=5e-324),
        KnownDistance("c", "d", 1, weight=1e-300, sigma_m=1e308),
        KnownDistance("d", "e", 1),
    ]
    with np.errstate(over="raise", invalid="raise"):
        rows, _ = _groups.distance_rows(deltas, first, second, distances, 4)
        expected = np.array(
            [
                _groups.distance_row(delta, a, b, distance, 4)[0]
                for delta, a, b, distance in zip(
                    deltas, first, second, distances, strict=True
                )
            ]
        )
    np.testing.assert_allclose(rows, expected, rtol=1e-14, atol=0)
    assert rows[0, 1] != 0
    assert rows[2, 0] != 0
    assert np.isinf(rows[3, 0])


@pytest.mark.parametrize("bad_index", [0, 2])
def test_distance_rows_report_first_coincident_pair_before_later_overflow(bad_index):
    deltas = np.ones((4, 3))
    deltas[bad_index] = 0
    deltas[-1] = [1.5e308, 1.5e308, 0]
    means = np.zeros((4, 3, 2))
    distances = [KnownDistance(str(index), str(index + 1), 1) for index in range(4)]
    with pytest.raises(
        ValueError, match=f"between '{bad_index}' and '{bad_index + 1}'.*coincident"
    ):
        _groups.distance_rows(deltas, means, means, distances, 4)


def test_distance_rows_reject_first_overflowing_length_before_later_coincidence():
    deltas = np.ones((3, 3))
    deltas[0] = [1.5e308, 1.5e308, 0]
    deltas[1] = 0
    means = np.zeros((3, 3, 2))
    distances = [KnownDistance(str(index), str(index + 1), 1) for index in range(3)]
    with pytest.raises(ValueError, match="contact distance exceeds"):
        _groups.distance_rows(deltas, means, means, distances, 4)


def test_distance_rows_preserve_empty_matrix_shape():
    rows, lengths = _groups.distance_rows(
        np.empty((0, 3)), np.empty((0, 3, 5)), np.empty((0, 3, 5)), [], 4
    )
    assert rows.shape == (0, 5)
    assert lengths.shape == (0,)
