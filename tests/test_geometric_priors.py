"""Prior rows retain their units and finite weighted differences."""

import json

import numpy as np
import pytest

from point_zero_cal import ContactGroup, GeometricCalibrator, GeometricParameter


class ParameterBackend:
    dof = 1
    joint_names = ("joint",)

    def __init__(self, parameters):
        self.geometric_parameters = tuple(parameters)


@pytest.mark.parametrize("count", [1, 3, 18, 36])
def test_prior_rows_match_individual_active_parameter_weights(count):
    parameters = [
        GeometricParameter(f"joint{i}", "x", 0.01 + i * 0.005, nominal=-0.1 + i * 0.01)
        for i in range(count)
    ]
    solver = GeometricCalibrator(ParameterBackend(parameters))
    active = np.arange(0, count, 2)
    values = np.linspace(-0.2, 0.2, count)
    rng = np.random.default_rng(127)
    matrix, residual = rng.normal(size=(12, len(active))), rng.normal(size=12)
    saved_matrix, saved_residual = matrix.copy(), residual.copy()
    augmented_matrix, augmented_residual = solver._with_prior_rows(
        values, active, 4, matrix, residual
    )
    scales = np.array([np.sqrt(4) / parameters[k].prior_scale for k in active])
    expected_residual = np.array(
        [
            (values[k] - parameters[k].nominal) * scales[row]
            for row, k in enumerate(active)
        ]
    )
    np.testing.assert_array_equal(augmented_matrix[:12], matrix)
    np.testing.assert_array_equal(augmented_matrix[12:], np.diag(scales))
    np.testing.assert_array_equal(augmented_residual[:12], residual)
    np.testing.assert_array_equal(augmented_residual[12:], expected_residual)
    np.testing.assert_array_equal(matrix, saved_matrix)
    np.testing.assert_array_equal(residual, saved_residual)


@pytest.mark.parametrize("count", [1, 3, 18])
@pytest.mark.parametrize("sign", [-1, 1])
def test_overflowing_parameter_difference_has_finite_weighted_prior(count, sign):
    parameters = [
        GeometricParameter(f"joint{i}", "x", 1e308, nominal=-sign * 1e308)
        for i in range(count)
    ]
    solver = GeometricCalibrator(ParameterBackend(parameters))
    values, active = np.full(count, sign * 1e308), np.arange(count)
    with np.errstate(over="raise", invalid="raise"):
        matrix, residual = solver._with_prior_rows(
            values, active, 2, np.zeros((6, count)), np.zeros(6)
        )
    np.testing.assert_allclose(
        matrix[6:], np.eye(count) * (np.sqrt(2) / 1e308), rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(
        residual[6:], np.full(count, sign * 2 * np.sqrt(2)), rtol=1e-14
    )


def test_extreme_finite_prior_reaches_the_analytical_regularized_solution():
    class Backend(ParameterBackend):
        def position_and_geometric_jacobian(self, q, values):
            return np.array([q[0] * (values[0] * 1e-308), 0, 0]), np.array(
                [[q[0] * 1e-308], [0], [0]]
            )

    solver = GeometricCalibrator(
        Backend([GeometricParameter("joint", "x", 1e308, nominal=-1e308)]),
        damping=0,
        step_tolerance=1e294,
        residual_tolerance=1,
    )
    with np.errstate(over="raise", invalid="raise"):
        fit = solver.calibrate_groups(
            [ContactGroup("a", [[-1], [1]])], initial_parameters=[1e308]
        )
    assert fit.converged
    assert fit.data_rank == 1
    assert fit.parameters[0].estimate == pytest.approx(-5e307, rel=1e-14)
    assert fit.evaluation.point_rms_m == pytest.approx(0.5, rel=1e-14)


def test_stalled_unbounded_fit_preserves_finite_prior_report_and_strict_json():
    class Backend(ParameterBackend):
        def position_and_geometric_jacobian(self, q, values):
            return np.array([q[0] * (values[0] * 1e-308), 0, 0]), np.array(
                [[q[0] * 1e-308], [0], [0]]
            )

    solver = GeometricCalibrator(
        Backend([GeometricParameter("joint", "x", 1e308, nominal=-1e308)]),
        residual_tolerance=1,
        step_tolerance=1e294,
    )
    with np.errstate(over="raise", invalid="raise"):
        fit = solver.calibrate_groups(
            [ContactGroup("a", [[-1], [1]])], initial_parameters=[1e308]
        )
    assert not fit.converged
    assert fit.message == "no residual-decreasing parameter step found"
    assert fit.prior_penalty == pytest.approx(2)
    json.dumps(fit.to_dict(), allow_nan=False)


def test_stalled_fit_reports_an_actual_blocking_bound():
    class Backend(ParameterBackend):
        def position_and_geometric_jacobian(self, q, values):
            return np.array([q[0] * (values[0] - 1), 0, 0]), np.array(
                [[q[0]], [0], [0]]
            )

    solver = GeometricCalibrator(
        Backend([GeometricParameter("joint", "x", 1e6, upper=0)]), residual_tolerance=2
    )
    fit = solver.calibrate_groups([ContactGroup("a", [[-1], [1]])])
    assert not fit.converged
    assert fit.message == "parameter bound blocked the remaining correction"
    assert fit.parameters[0].estimate == 0


@pytest.mark.parametrize("count", [1, 4])
def test_unrepresentable_prior_matrix_is_rejected_with_a_value_error(count):
    class Backend(ParameterBackend):
        def position_and_geometric_jacobian(self, q, values):
            return np.zeros(3), np.zeros((3, count))

    solver = GeometricCalibrator(
        Backend([GeometricParameter(f"joint{i}", "x", 5e-324) for i in range(count)])
    )
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="weighted contact residuals"):
            solver.calibrate_groups([ContactGroup("a", [[-1], [1]])])
