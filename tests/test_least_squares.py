"""Numerical extremes and backtracking behavior of the shared solver."""

import decimal
from types import SimpleNamespace

import numpy as np
import pytest

from point_zero_cal._least_squares import (
    _scaled_left_projection,
    _scaled_right_product,
    backtracking_solve,
    damped_svd_step,
    numerical_rank,
)


def test_damping_preserves_representable_step_at_tiny_length_scale():
    step, remaining, _ = damped_svd_step(
        np.array([[1e-200]]), np.array([1.0]), damping=1.0, rank_tolerance=1e-8
    )
    np.testing.assert_allclose(step, [1e-200], rtol=1e-14, atol=0)
    assert remaining == 1e200


@pytest.mark.parametrize("damping, expected", [(0.0, 1e208), (1e200, 8e207)])
def test_overflowing_projection_preserves_finite_correction(damping, expected):
    with np.errstate(over="raise", invalid="raise"):
        step, remaining, values = damped_svd_step(
            np.full((4, 1), 1e100),
            np.full(4, 1e308),
            damping=damping,
            rank_tolerance=1e-8,
        )
    np.testing.assert_allclose(step, [expected], rtol=1e-14)
    assert remaining == pytest.approx(1e208)
    np.testing.assert_allclose(values, [2e100])


def test_projection_fallback_preserves_tiny_and_large_corrections_together():
    matrix = np.array([[1e100, 0], [1e100, 0], [0, 1e308]])
    error = np.array([1.4e308, 1.4e308, 1.4e292])
    step, remaining, _ = damped_svd_step(
        matrix,
        error,
        damping=0,
        rank_tolerance=1e-250,
    )
    np.testing.assert_allclose(step, [1.4e208, 1.4e-16], rtol=1e-14, atol=0)
    assert remaining == pytest.approx(1.4e208)


def test_projection_fallback_preserves_already_finite_modes():
    matrix = np.array([[1e100, 0], [1e100, 0], [0, 1e-200]])
    error = np.array([1.4e308, 1.4e308, 1e-100])
    step, remaining, _ = damped_svd_step(
        matrix,
        error,
        damping=0,
        rank_tolerance=1e-310,
    )
    np.testing.assert_allclose(step, [1.4e208, 1e100], rtol=1e-14, atol=0)
    assert remaining == pytest.approx(1.4e208)


@pytest.mark.parametrize("damping", [0.0, 1.0])
def test_overflowing_projection_retains_a_tiny_remainder_after_cancellation(damping):
    count = 16384
    error = np.r_[
        0.0,
        np.full(count // 2 - 1, 1.7e308),
        np.full(count // 2 - 1, -1.7e308),
        1e-100,
    ]
    with np.errstate(over="raise", invalid="raise"):
        step, remaining, _ = damped_svd_step(
            np.ones((count, 1)), error, damping=damping, rank_tolerance=1e-8
        )
    np.testing.assert_allclose(step, [1e-100 / (count + damping)], rtol=1e-14, atol=0)
    assert remaining == pytest.approx(1e-100 / count, rel=1e-14, abs=0)


@pytest.mark.parametrize("repeats", [1, 32])
def test_scaled_left_projection_preserves_extreme_modes_and_decimal_context(repeats):
    left = np.zeros((2 * repeats + 1, 2))
    left[:-1, 0] = 0.5 / repeats
    left[-1, 1] = 1
    error = np.r_[np.full(2 * repeats, 1.7e308), 1e-308]
    with decimal.localcontext() as context:
        context.prec = 3
        context.Emax = 20
        context.Emin = -20
        context.traps[decimal.Inexact] = True
        context.traps[decimal.Rounded] = True
        before = context.copy()
        mantissas, exponents = _scaled_left_projection(left, error)
        assert context.flags == before.flags
        assert context.traps == before.traps
    np.testing.assert_array_equal(np.ldexp(mantissas, exponents), [1.7e308, 1e-308])


@pytest.mark.parametrize("damping", [0.0, 1.0])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("coefficient", [1e-300, 1e-100])
def test_projection_underflow_keeps_a_representable_undamped_correction(
    damping, mixed, coefficient
):
    count = 1024
    matrix = np.full((count, 1), coefficient)
    error = np.zeros(count)
    error[0] = np.nextafter(0.0, 1.0)
    expected = float(np.longdouble(error[0]) / (count * np.longdouble(coefficient)))
    expected_step = np.array([expected if damping == 0 else 0])
    if mixed:
        matrix = np.column_stack((np.zeros(count), matrix[:, 0]))
        matrix = np.vstack(([1, 0], matrix))
        error = np.r_[7.0, error]
        expected_step = np.r_[7 / (1 + damping), expected_step]
    with np.errstate(over="raise", invalid="raise", under="raise"):
        step, remaining, _ = damped_svd_step(
            matrix, error, damping=damping, rank_tolerance=1e-310
        )
    np.testing.assert_allclose(step, expected_step, rtol=1e-14, atol=0)
    assert remaining == pytest.approx(7 if mixed else expected, rel=1e-14, abs=0)


def test_normal_scale_jacobian_keeps_a_subnormal_parameter_step():
    smallest = np.nextafter(0.0, 1.0)
    with np.errstate(over="raise", invalid="raise", under="raise"):
        step, remaining, _ = damped_svd_step(
            np.ones((4, 1)), np.full(4, smallest), damping=0, rank_tolerance=1e-8
        )
    np.testing.assert_array_equal(step, [smallest])
    assert remaining == smallest


def test_orthogonal_nonzero_residual_has_a_finite_zero_correction():
    with np.errstate(all="raise"):
        step, remaining, _ = damped_svd_step(
            np.array([[1.0], [0.0]]),
            np.array([0.0, 1.0]),
            damping=0,
            rank_tolerance=1e-8,
        )
    np.testing.assert_array_equal(step, [0.0])
    assert remaining == 0


@pytest.mark.parametrize("large_error", [1e100, 1e200])
def test_small_singular_direction_retains_a_tiny_residual(large_error):
    matrix = np.array([[1e100, 0], [1e100, 0], [0, 1e-200]])
    error = np.array([large_error, large_error, 1e-100])
    step, remaining, _ = damped_svd_step(
        matrix, error, damping=0, rank_tolerance=1e-310
    )
    np.testing.assert_allclose(step, [large_error / 1e100, 1e100], rtol=1e-14, atol=0)
    assert remaining == pytest.approx(1e100)


@pytest.mark.parametrize("damping, expected", [(0.0, 1e308), (1.0, 4e-308)])
def test_unrepresentable_singular_coefficient_can_map_to_finite_parameters(
    damping, expected
):
    step, remaining, _ = damped_svd_step(
        np.full((1, 4), 1e-308),
        np.array([4.0]),
        damping=damping,
        rank_tolerance=1e-8,
    )
    np.testing.assert_allclose(step, [expected] * 4, rtol=1e-14, atol=0)
    assert remaining == pytest.approx(1e308)


def test_scaled_projection_preserves_tiny_mode_after_large_cancellation():
    right = np.array(
        [
            [0.25] * 16,
            [0.25] * 8 + [-0.25] * 8,
            [0.25, -0.25] * 8,
        ]
    )
    result = _scaled_right_product(
        right,
        np.array([0.75, 0.75, 0.5]),
        np.array([1025, 1025, -1022], dtype=np.int32),
    )
    np.testing.assert_array_equal(result[:8], np.ldexp(0.75, 1024))
    np.testing.assert_array_equal(
        result[8:], np.array([1, -1] * 4) * np.ldexp(0.125, -1022)
    )


@pytest.mark.parametrize("damping, expected", [(0.0, 1e308), (1.0, 4e-308)])
def test_svd_fallback_is_independent_of_decimal_context(damping, expected):
    with decimal.localcontext() as context:
        context.prec = 3
        context.Emax = 20
        context.Emin = -20
        context.rounding = decimal.ROUND_DOWN
        context.clamp = 1
        context.traps[decimal.Inexact] = True
        context.traps[decimal.Rounded] = True
        context.traps[decimal.FloatOperation] = True
        context.flags[decimal.Inexact] = True
        before = context.copy()
        step, remaining, _ = damped_svd_step(
            np.full((1, 4), 1e-308),
            np.array([4.0]),
            damping=damping,
            rank_tolerance=1e-8,
        )
        np.testing.assert_allclose(step, [expected] * 4, rtol=1e-14, atol=0)
        assert remaining == pytest.approx(1e308)
        assert context.flags == before.flags
        assert context.traps == before.traps
        assert (
            context.prec,
            context.Emax,
            context.Emin,
            context.rounding,
            context.clamp,
        ) == (before.prec, before.Emax, before.Emin, before.rounding, before.clamp)


def test_unrepresentable_correction_still_raises():
    with pytest.raises(ValueError, match="non-finite offset step"):
        damped_svd_step(
            np.array([[1e-200]]),
            np.array([1e200]),
            damping=0,
            rank_tolerance=1e-8,
        )


def test_unrepresentable_spectrum_raises_instead_of_reporting_zero_rank():
    with pytest.raises(ValueError, match="spectrum exceeds"):
        damped_svd_step(
            np.full((4, 1), 1e308),
            np.ones(4),
            damping=0,
            rank_tolerance=1e-8,
        )


@pytest.mark.parametrize("tolerance, rank", [(0.8, 1), (0.4, 2)])
def test_subnormal_spectrum_uses_relative_rank(tolerance, rank):
    smallest = np.nextafter(0.0, 1.0)
    values = np.array([2 * smallest, smallest])
    assert numerical_rank(values, tolerance) == rank
    step, remaining, _ = damped_svd_step(
        np.diag(values),
        values.copy(),
        damping=0,
        rank_tolerance=tolerance,
    )
    np.testing.assert_array_equal(step, [1, 0] if rank == 1 else [1, 1])
    assert remaining == 1


@pytest.mark.parametrize("values", [np.array([]), np.zeros(3)])
def test_empty_and_zero_spectra_have_zero_rank(values):
    assert numerical_rank(values, 1e-8) == 0


@pytest.mark.parametrize("shape", [(20, 5), (3, 5)])
@pytest.mark.parametrize("rank", [0, 1, 3])
@pytest.mark.parametrize("damping", [0.0, 0.3])
def test_svd_step_matches_truncated_pseudoinverse(shape, rank, damping):
    rng = np.random.default_rng(105)
    left, _ = np.linalg.qr(rng.normal(size=(shape[0], min(shape))))
    right, _ = np.linalg.qr(rng.normal(size=(shape[1], min(shape))))
    values = np.zeros(min(shape))
    values[:rank] = np.arange(rank, 0, -1)
    matrix = (left * values) @ right.T
    error = rng.normal(size=shape[0])
    expected_remaining = right[:, :rank] @ ((left[:, :rank].T @ error) / values[:rank])
    expected_step = right[:, :rank] @ (
        (left[:, :rank].T @ error) * values[:rank] / (values[:rank] ** 2 + damping)
    )
    step, remaining, spectrum = damped_svd_step(
        matrix,
        error,
        damping=damping,
        rank_tolerance=1e-8,
    )
    np.testing.assert_allclose(step, expected_step, rtol=1e-12, atol=1e-14)
    assert remaining == pytest.approx(np.linalg.norm(expected_remaining, ord=np.inf))
    np.testing.assert_allclose(spectrum, values, rtol=1e-12, atol=1e-14)


def test_bound_blocked_step_does_not_reevaluate_identical_parameters():
    calls = []

    def evaluate(x):
        calls.append(x.copy())
        return SimpleNamespace(matrix=np.ones((1, 1)), error=1 - x, score=1 - x[0])

    fit = backtracking_solve(
        np.zeros(1),
        evaluate,
        lambda x, step: np.minimum(x + step, 0),
        max_iterations=10,
        step_tolerance=1e-8,
        damping=0,
        rank_tolerance=1e-8,
    )
    assert fit.stalled
    assert len(calls) == 1


def test_repeated_projected_trials_still_allow_shorter_successful_step():
    calls = []

    def evaluate(x):
        calls.append(float(x[0]))
        return SimpleNamespace(
            matrix=np.ones((1, 1)), error=np.array([8.0]), score=(x[0] - 0.5) ** 2
        )

    fit = backtracking_solve(
        np.zeros(1),
        evaluate,
        lambda x, step: np.minimum(x + step, 2),
        max_iterations=1,
        step_tolerance=1e-8,
        damping=0,
        rank_tolerance=1e-8,
    )
    assert calls == [0.0, 2.0, 1.0, 0.5]
    np.testing.assert_array_equal(fit.parameters, [0.5])
