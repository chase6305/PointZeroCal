"""Numerical extremes and backtracking behavior of the shared solver."""

from types import SimpleNamespace

import numpy as np

from point_zero_cal._least_squares import backtracking_solve, damped_svd_step


def test_damping_preserves_representable_step_at_tiny_length_scale():
    step, remaining, _ = damped_svd_step(
        np.array([[1e-200]]), np.array([1.0]), damping=1.0, rank_tolerance=1e-8
    )
    np.testing.assert_allclose(step, [1e-200], rtol=1e-14, atol=0)
    assert remaining == 1e200


def test_bound_blocked_step_does_not_reevaluate_identical_parameters():
    calls = []

    def evaluate(x):
        calls.append(x.copy())
        return SimpleNamespace(matrix=np.ones((1, 1)), error=1 - x, score=1 - x[0])

    fit = backtracking_solve(
        np.zeros(1), evaluate, lambda x, step: np.minimum(x + step, 0),
        max_iterations=10, step_tolerance=1e-8, damping=0, rank_tolerance=1e-8,
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
        np.zeros(1), evaluate, lambda x, step: np.minimum(x + step, 2),
        max_iterations=1, step_tolerance=1e-8, damping=0, rank_tolerance=1e-8,
    )
    assert calls == [0.0, 2.0, 1.0, 0.5]
    np.testing.assert_array_equal(fit.parameters, [0.5])
