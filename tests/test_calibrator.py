import json

import numpy as np
import pytest

from point_zero_cal import (
    ContactGroup,
    JointZeroCalibrator,
    KnownDistance,
    SampleDiagnostics,
    _groups,
)


class PlanarArm:
    dof = 2
    joint_names = ("shoulder", "elbow")

    def position_and_jacobian(self, q):
        q1, q2 = q
        position = np.array(
            [np.cos(q1) + np.cos(q1 + q2), np.sin(q1) + np.sin(q1 + q2), 0.0]
        )
        jacobian = np.array(
            [
                [-np.sin(q1) - np.sin(q1 + q2), -np.sin(q1 + q2)],
                [np.cos(q1) + np.cos(q1 + q2), np.cos(q1 + q2)],
                [0.0, 0.0],
            ]
        )
        return position, jacobian


def test_recovers_offsets_from_fixed_point_samples():
    expected = np.array([0.0, -0.025])
    contact_configurations = np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]])
    measured = contact_configurations - expected

    result = JointZeroCalibrator(PlanarArm()).calibrate(measured)

    assert result.converged
    np.testing.assert_allclose(result.offsets_rad, expected, atol=1e-7)
    assert result.rank == 1
    assert result.message == "calibration converged"
    assert result.sample_count == 2
    assert result.active_joint_names == ("elbow",)
    assert result.corrected_point_rms < result.initial_point_rms
    assert result.residual_rms == result.corrected_point_rms


def test_numpy_joint_names_support_fixed_names_and_json_reports():
    class Backend(PlanarArm):
        joint_names = np.array(PlanarArm.joint_names)

    expected = np.array([0.0, -0.025])
    q = np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]])
    result = JointZeroCalibrator(Backend()).calibrate(
        q - expected, fixed_joints=["shoulder"]
    )
    assert result.converged
    assert result.active_joint_names == ("elbow",)
    np.testing.assert_allclose(result.offsets_rad, expected, atol=1e-7)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "samples",
    [np.zeros(2), np.zeros((1, 2)), np.zeros((2, 3)), [[0.0, np.nan], [0.0, 0.0]]],
)
def test_rejects_invalid_samples(samples):
    with pytest.raises(ValueError, match="samples"):
        JointZeroCalibrator(PlanarArm()).calibrate(samples)


def test_accepts_degrees_and_fixed_joint_names():
    samples = np.rad2deg(np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]]))
    result = JointZeroCalibrator(PlanarArm()).calibrate(
        samples, sample_unit="deg", fixed_joints=["shoulder"]
    )
    assert result.offsets_rad[0] == 0.0


def test_initial_rms_is_measured_from_initial_offset_seed():
    samples = np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]])
    seed = np.array([0.0, -0.01])
    arm = PlanarArm()
    seeded_positions = np.vstack(
        [arm.position_and_jacobian(q + seed)[0] for q in samples]
    )
    centered = seeded_positions - seeded_positions.mean(axis=0)
    expected_rms = np.sqrt(np.mean(np.sum(centered**2, axis=1)))

    result = JointZeroCalibrator(arm).calibrate(samples, initial_offsets=seed)

    assert result.initial_point_rms == pytest.approx(expected_rms)


def test_rank_deficient_problem_does_not_report_convergence():
    samples = np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]])
    result = JointZeroCalibrator(PlanarArm(), max_iterations=3).calibrate(
        samples, fixed_joints=[]
    )
    assert not result.converged
    assert result.rank < 2
    assert result.message.startswith("insufficient observability")


def test_rejects_unknown_fixed_joint():
    with pytest.raises(ValueError, match="unknown joint"):
        JointZeroCalibrator(PlanarArm()).calibrate(
            np.zeros((2, 2)), fixed_joints=["missing"]
        )


def test_rejects_invalid_backend_output_shape():
    class InvalidBackend(PlanarArm):
        def position_and_jacobian(self, q):
            return np.zeros(2), np.zeros((2, 2))

    with pytest.raises(ValueError, match="position shape"):
        JointZeroCalibrator(InvalidBackend()).calibrate(np.zeros((2, 2)))


def test_backend_may_reuse_position_and_jacobian_buffers():
    class BufferedArm(PlanarArm):
        def __init__(self):
            self.position = np.empty(3)
            self.jacobian = np.empty((3, 2))

        def position_and_jacobian(self, q):
            self.position[:], self.jacobian[:] = super().position_and_jacobian(q)
            return self.position, self.jacobian

    expected = np.array([0.0, -0.025])
    measured = np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]]) - expected
    result = JointZeroCalibrator(BufferedArm()).calibrate(measured)

    assert result.converged
    np.testing.assert_allclose(result.offsets_rad, expected, atol=1e-8)


class SineKinematics:
    dof = 1
    joint_names = ("joint",)

    def position_and_jacobian(self, q):
        return np.array([np.sin(q[0]), 0.0, 0.0]), np.array([[np.cos(q[0])], [0], [0]])


def test_single_state_reuses_centred_position_residual(monkeypatch):
    solver = JointZeroCalibrator(PlanarArm())
    q = np.array([[0.0, 1.2], [0.8, -1.3], [1.1, -1.5]])
    positions, _ = solver._evaluate(q)
    expected_score = solver._point_rms(positions)
    center = _groups.center
    calls = []

    def counted_center(values):
        calls.append(values.shape)
        return center(values)

    monkeypatch.setattr(_groups, "center", counted_center)
    state = solver._single_state(q, np.array([1]))
    assert calls == [(3, 3), (3, 3, 1)]
    assert state.score == expected_score
    np.testing.assert_allclose(state.error.reshape(-1, 3), -center(positions))


def test_observability_report_preserves_a_subnormal_observable_direction():
    solver = JointZeroCalibrator(SineKinematics(), rank_tolerance=0.8)
    report = solver._observability_report(np.array([[1e-323]]), np.array([0]))
    assert report.rank == 1
    assert report.fully_observable
    assert report.condition_number == 1


def test_observability_normalization_does_not_erase_a_representable_spectrum():
    smallest = np.nextafter(0.0, 1.0)
    matrix = np.array([[smallest], [-smallest], [0], [0], [0], [0]])
    report = JointZeroCalibrator(SineKinematics())._observability_report(
        matrix, np.array([0])
    )
    assert report.rank == 1
    assert report.condition_number == 1
    np.testing.assert_array_equal(report.singular_values, [smallest])


@pytest.mark.parametrize("strong", [1e-15, 0.5, 2.0])
@pytest.mark.parametrize("rows", [6, 32])
def test_observability_keeps_a_subnormal_mode_beside_a_strong_column(strong, rows):
    smallest = np.nextafter(0.0, 1.0)
    matrix = np.zeros((rows, 2))
    matrix[:2, 0] = [strong, -strong]
    matrix[2:, 1] = np.tile([smallest, -smallest], (rows - 2) // 2)
    solver = JointZeroCalibrator(PlanarArm(), rank_tolerance=1e-310)
    report = solver._observability_report(matrix, np.arange(2))
    np.testing.assert_allclose(report.singular_values[0], strong * np.sqrt(2 / rows))
    assert report.singular_values[1] == smallest
    assert report.rank == (2 if strong == 1e-15 else 1)


def test_extreme_independent_groups_support_fitting_evaluation_and_ranking():
    class SeparatedContacts:
        dof = 2
        joint_names = ("reference", "offset")

        def position_and_jacobian(self, q):
            return (
                np.array([q[0] * 1e308, np.sin(q[1]), 0]),
                np.array([[1e308, 0], [0, np.cos(q[1])], [0, 0]]),
            )

    groups = [
        ContactGroup("a", [[-1, 0.02], [-1, np.pi + 0.02]], sigma_m=0.5),
        ContactGroup("b", [[1, 0.02], [1, np.pi + 0.02]]),
    ]
    solver = JointZeroCalibrator(SeparatedContacts())
    with np.errstate(over="raise", invalid="raise"):
        result = solver.calibrate_groups(groups)
        evaluation = solver.evaluate_groups(groups, offsets_rad=result.offsets_rad)
        ranking = solver.rank_contact_candidates(
            groups, [[-1, 2 * np.pi]], group_name="a"
        )
    assert result.converged
    assert result.offsets_rad[1] == pytest.approx(-0.02, abs=1e-8)
    assert evaluation.point_rms_m < 1e-8
    assert ranking.baseline.rank == ranking.recommended.observability.rank == 1


@pytest.mark.parametrize("action", ["fit", "evaluate", "rank"])
def test_unrepresentable_contact_distance_is_rejected(action):
    class Backend:
        dof = 1
        joint_names = ("joint",)

        def position_and_jacobian(self, q):
            position = np.array([1.5e308, 1.5e308, 0]) if q[0] < 0 else np.zeros(3)
            return position, np.zeros((3, 1))

    solver = JointZeroCalibrator(Backend())
    groups = [ContactGroup("a", [[-1], [-1]]), ContactGroup("b", [[1], [1]])]
    distances = [KnownDistance("a", "b", 1)]
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="contact distance exceeds"):
            if action == "fit":
                solver.calibrate_groups(groups, distances=distances, fixed_joints=[])
            elif action == "evaluate":
                solver.evaluate_groups(groups, offsets_rad=[0], distances=distances)
            else:
                solver.rank_contact_candidates(
                    groups, [[-1]], group_name="a", distances=distances, fixed_joints=[]
                )


@pytest.mark.parametrize("damping", [0.0, 1e-10])
def test_backtracking_handles_an_overshooting_initial_step(damping):
    # Full Gauss-Newton step from this seed increases the position residual.
    seed = 1.41
    assert abs(np.sin(seed - np.tan(seed))) > abs(np.sin(seed))
    samples = [[0.0], [np.pi]]
    kwargs = {"fixed_joints": [], "initial_offsets": [seed]}

    first = JointZeroCalibrator(
        SineKinematics(), max_iterations=1, damping=damping
    ).calibrate(samples, **kwargs)
    result = JointZeroCalibrator(SineKinematics(), damping=damping).calibrate(
        samples, **kwargs
    )

    assert first.corrected_point_rms < first.initial_point_rms
    assert not first.converged
    assert result.converged
    np.testing.assert_allclose(result.offsets_rad, 0.0, atol=1e-8)


def test_final_small_step_can_satisfy_a_tight_residual_tolerance():
    result = JointZeroCalibrator(SineKinematics(), residual_tolerance=1e-12).calibrate(
        [[0.0], [np.pi]], fixed_joints=[], initial_offsets=[1.41]
    )

    assert result.converged
    assert result.corrected_point_rms < 1e-12


def test_stationary_noisy_fit_stops_without_claiming_convergence():
    samples = np.array([[0.0, np.pi / 2], [np.pi / 2 + 0.01, -np.pi / 2]])
    result = JointZeroCalibrator(PlanarArm()).calibrate(samples)

    assert not result.converged
    assert result.iterations < 10
    assert result.message == "point residual did not reach tolerance"
    assert 1e-6 < result.corrected_point_rms <= result.initial_point_rms


def test_unobservable_directions_are_excluded_even_without_damping():
    class WeakDirection:
        dof = 3
        joint_names = ("reference", "strong", "weak")

        def position_and_jacobian(self, q):
            sign, strong, weak = q
            return np.array(
                [sign * (strong - 0.1), sign * 1e-12 * (weak - 1), 0]
            ), np.array(
                [
                    [strong - 0.1, sign, 0],
                    [1e-12 * (weak - 1), 0, sign * 1e-12],
                    [0, 0, 0],
                ]
            )

    result = JointZeroCalibrator(WeakDirection(), damping=0.0).calibrate(
        [[-1, 0, 0], [1, 0, 0]]
    )

    assert not result.converged
    assert result.rank == 1
    assert np.isinf(result.condition_number)
    np.testing.assert_allclose(result.offsets_rad, [0, 0.1, 0], atol=1e-14)


def test_underdetermined_system_has_infinite_condition_number():
    class WideBackend:
        dof = 8
        joint_names = tuple(f"joint{i}" for i in range(dof))
        jacobians = np.random.default_rng(42).normal(size=(2, 3, dof))

        def position_and_jacobian(self, q):
            return np.zeros(3), self.jacobians[int(q[0])]

    samples = np.zeros((2, 8))
    samples[1, 0] = 1
    result = JointZeroCalibrator(WideBackend(), rank_tolerance=1e-20).calibrate(
        samples, fixed_joints=[]
    )

    assert not result.converged
    assert result.rank < 8
    assert np.isinf(result.condition_number)


def test_preserves_inputs_fixed_offsets_and_read_only_results():
    expected = np.array([0.2, -0.025])
    samples = np.array([[0.0, np.pi / 2], [np.pi / 2, -np.pi / 2]]) - expected
    original = samples.copy()
    seed = np.array([0.2, 0.0])
    result = JointZeroCalibrator(PlanarArm()).calibrate(samples, initial_offsets=seed)

    assert result.converged
    np.testing.assert_array_equal(samples, original)
    np.testing.assert_array_equal(seed, [0.2, 0.0])
    np.testing.assert_allclose(result.offsets_rad, expected, atol=1e-8)
    with pytest.raises(ValueError, match="read-only"):
        result.offsets_rad[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        result.singular_values[0] = 0


@pytest.mark.parametrize(
    "options",
    [{"max_iterations": value} for value in (0, -1, 1.5, np.nan, True)]
    + [
        {name: value}
        for name in ("step_tolerance", "residual_tolerance", "rank_tolerance")
        for value in (0, -1, np.nan, np.inf, True)
    ]
    + [{"damping": value} for value in (-1, np.nan, np.inf, True)]
    + [{"rank_tolerance": 1}],
)
def test_rejects_invalid_solver_options(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        JointZeroCalibrator(PlanarArm(), **options)


@pytest.mark.parametrize("fixed", [[-1], [2], [True], [0.5], "shoulder", [0, 1]])
def test_rejects_invalid_fixed_joint_selection(fixed):
    with pytest.raises(ValueError, match="joint"):
        JointZeroCalibrator(PlanarArm()).calibrate(np.zeros((2, 2)), fixed_joints=fixed)


@pytest.mark.parametrize("seed", [[0], [0, np.inf], [[0, 0]]])
def test_rejects_invalid_initial_offsets(seed):
    with pytest.raises(ValueError, match="initial_offsets"):
        JointZeroCalibrator(PlanarArm()).calibrate(
            np.zeros((2, 2)), initial_offsets=seed
        )


def test_rejects_invalid_sample_unit():
    with pytest.raises(ValueError, match="sample_unit"):
        JointZeroCalibrator(PlanarArm()).calibrate(
            np.zeros((2, 2)), sample_unit="turns"
        )


@pytest.mark.parametrize("field", ["position", "jacobian"])
def test_rejects_nonfinite_backend_outputs(field):
    class InvalidBackend(PlanarArm):
        def position_and_jacobian(self, q):
            position, jacobian = super().position_and_jacobian(q)
            if field == "position":
                position[0] = np.inf
            else:
                jacobian[0, 0] = np.nan
            return position, jacobian

    with pytest.raises(ValueError, match="kinematics returned non-finite"):
        JointZeroCalibrator(InvalidBackend()).calibrate(np.zeros((2, 2)))


@pytest.mark.parametrize("dof, names", [(0, ()), (2, ("a",)), (2, ("a", "a"))])
def test_rejects_invalid_backend_metadata(dof, names):
    backend = PlanarArm()
    backend.dof, backend.joint_names = dof, names
    with pytest.raises(ValueError, match="kinematics"):
        JointZeroCalibrator(backend)


@pytest.mark.parametrize("count", [3, 10, 41, 1000])
def test_identical_poses_have_zero_rank_and_zero_dispersion(count):
    samples = np.tile([0.3, 0.7], (count, 1))
    result = JointZeroCalibrator(PlanarArm()).calibrate(samples)

    assert not result.converged
    assert result.rank == 0
    assert result.initial_point_rms == result.corrected_point_rms == 0
    assert np.isinf(result.condition_number)
    np.testing.assert_array_equal(result.singular_values, 0)


@pytest.mark.parametrize("scale", [1e-170, 1.0, 1e170])
def test_calibration_rms_and_offsets_survive_extreme_length_scales(scale):
    class ScaledSine(SineKinematics):
        def position_and_jacobian(self, q):
            position, jacobian = super().position_and_jacobian(q)
            return scale * position, scale * jacobian

    result = JointZeroCalibrator(
        ScaledSine(), damping=0, residual_tolerance=scale * 1e-8
    ).calibrate([[0.0], [np.pi]], fixed_joints=[], initial_offsets=[0.1])

    assert result.converged
    assert result.initial_point_rms / scale == pytest.approx(np.sin(0.1))
    assert result.corrected_point_rms / scale < 1e-8
    np.testing.assert_allclose(result.offsets_rad, 0.0, atol=1e-8)


def test_large_damping_does_not_falsely_satisfy_step_tolerance():
    result = JointZeroCalibrator(
        SineKinematics(), damping=1e9, max_iterations=2, residual_tolerance=1.0
    ).calibrate([[0.0], [np.pi]], fixed_joints=[], initial_offsets=[0.1])

    assert not result.converged
    assert result.step_norm == pytest.approx(abs(np.tan(result.offsets_rad[0])))
    assert result.step_norm > 0.1


def test_solution_reached_on_last_update_reports_convergence():
    class QuadraticKinematics(SineKinematics):
        def position_and_jacobian(self, q):
            return np.array([q[0] ** 2, 0, 0]), np.array([[2 * q[0]], [0], [0]])

    result = JointZeroCalibrator(
        QuadraticKinematics(), damping=0, max_iterations=1
    ).calibrate([[-1.0], [1.0]], fixed_joints=[], initial_offsets=[0.1])

    assert result.converged
    assert result.iterations == 1
    assert result.step_norm < 1e-8
    assert result.corrected_point_rms < 1e-8
    np.testing.assert_allclose(result.offsets_rad, 0, atol=1e-8)


@pytest.mark.parametrize("translation", [0.0, 1e6])
def test_world_translation_and_sample_order_preserve_calibration(translation):
    class TranslatedSine(SineKinematics):
        def position_and_jacobian(self, q):
            position, jacobian = super().position_and_jacobian(q)
            return position + translation, jacobian

    samples = np.tile([[0.0], [np.pi]], (20, 1))
    solver = JointZeroCalibrator(TranslatedSine())
    results = [
        solver.calibrate(q, fixed_joints=[], initial_offsets=[0.1])
        for q in (samples, samples[np.random.default_rng(42).permutation(len(samples))])
    ]

    for result in results:
        assert result.converged
        assert result.initial_point_rms == pytest.approx(np.sin(0.1))
        np.testing.assert_allclose(result.offsets_rad, 0, atol=1e-8)
    np.testing.assert_allclose(
        results[0].offsets_rad, results[1].offsets_rad, atol=1e-10
    )


@pytest.mark.parametrize("field", ["samples", "initial_offsets"])
def test_rejects_complex_inputs_without_discarding_imaginary_values(field):
    samples = np.zeros((2, 2))
    options = {}
    if field == "samples":
        samples = samples.astype(complex) + 1j
    else:
        options[field] = np.array([0, 1j])
    with pytest.raises(ValueError, match=f"{field}.*real"):
        JointZeroCalibrator(PlanarArm()).calibrate(samples, **options)


@pytest.mark.parametrize("field", ["position", "jacobian"])
def test_rejects_complex_backend_outputs(field):
    class ComplexBackend(PlanarArm):
        def position_and_jacobian(self, q):
            position, jacobian = super().position_and_jacobian(q)
            if field == "position":
                position = position.astype(complex) + 1j
            else:
                jacobian = jacobian.astype(complex) + 1j
            return position, jacobian

    with pytest.raises(ValueError, match=f"kinematics {field}.*real"):
        JointZeroCalibrator(ComplexBackend()).calibrate(np.zeros((2, 2)))


def test_overflowing_corrected_configuration_is_rejected_before_backend_calls():
    class UncalledBackend(PlanarArm):
        def position_and_jacobian(self, q):
            raise AssertionError("non-finite configurations must not reach the backend")

    with pytest.raises(ValueError, match="corrected configurations.*finite"):
        JointZeroCalibrator(UncalledBackend()).calibrate(
            np.full((2, 2), 1e308), initial_offsets=[1e308, 1e308]
        )


def test_out_of_range_scalar_is_reported_as_value_error():
    with pytest.raises(ValueError, match="damping.*finite"):
        JointZeroCalibrator(PlanarArm(), damping=10**400)


def test_sample_diagnostics_match_seeded_and_corrected_forward_kinematics():
    arm = PlanarArm()
    samples = np.array([[0.0, np.pi / 2], [np.pi / 2 + 0.01, -np.pi / 2]])
    seed = np.array([0.2, -0.01])
    result = JointZeroCalibrator(arm).calibrate(samples, initial_offsets=seed)
    diagnostics = result.sample_diagnostics
    assert diagnostics is not None

    for offsets, distances, rms in (
        (seed, diagnostics.initial_point_errors, result.initial_point_rms),
        (
            result.offsets_rad,
            diagnostics.corrected_point_errors,
            result.corrected_point_rms,
        ),
    ):
        positions = np.array(
            [arm.position_and_jacobian(q + offsets)[0] for q in samples]
        )
        expected = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
        np.testing.assert_allclose(distances, expected, atol=1e-14)
        assert np.sqrt(np.mean(distances**2)) == pytest.approx(rms)
    np.testing.assert_allclose(
        diagnostics.contact_point_xyz, positions.mean(axis=0), atol=1e-14
    )
    assert diagnostics.max_corrected_point_error == max(
        diagnostics.corrected_point_errors
    )
    assert (
        diagnostics.corrected_point_errors[diagnostics.worst_sample_index]
        == diagnostics.max_corrected_point_error
    )
    for array in (
        diagnostics.contact_point_xyz,
        diagnostics.initial_point_errors,
        diagnostics.corrected_point_errors,
    ):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1.0


def test_result_serialization_is_strict_json_and_detached_from_result_arrays():
    result = JointZeroCalibrator(PlanarArm()).calibrate(np.zeros((3, 2)))
    payload = result.to_dict()
    restored = json.loads(json.dumps(payload, allow_nan=False))

    assert restored["condition_number"] is None
    assert restored["sample_diagnostics"]["corrected_point_errors_m"] == [0, 0, 0]
    payload["offsets_rad"][0] = 10
    payload["sample_diagnostics"]["corrected_point_errors_m"][0] = 10
    assert result.offsets_rad[0] == 0
    assert result.sample_diagnostics.corrected_point_errors[0] == 0


@pytest.mark.parametrize(
    "xyz, initial, corrected",
    [
        ([0, 0], [0, 0], [0, 0]),
        ([0, 0, 0], [], []),
        ([0, 0, 0], [0, 0], [0]),
        ([0, 0, 0], [0, -1], [0, 0]),
        ([0, 0, 0], [0, 0], [0, np.inf]),
        ([0, 0, 1j], [0, 0], [0, 0]),
    ],
)
def test_sample_diagnostics_reject_invalid_arrays(xyz, initial, corrected):
    with pytest.raises(ValueError):
        SampleDiagnostics(xyz, initial, corrected)
