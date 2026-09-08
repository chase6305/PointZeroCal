"""Regression tests for shared contact reductions and weighted rows."""

import numpy as np
import pytest

from point_zero_cal import ContactGroup, KnownDistance
from point_zero_cal import _groups


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
        [KnownDistance("a", "b", 0.5, weight=1e308)], "rad", 1,
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
        [], "rad", 1,
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
    for group, (rms, errors) in zip(report.groups, expected):
        assert group.point_rms_m == rms
        np.testing.assert_array_equal(group.point_errors_m, errors)
