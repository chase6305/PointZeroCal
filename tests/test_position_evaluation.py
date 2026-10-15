"""Optional position-only backends preserve held-out report contracts."""

import numpy as np
import pytest

from point_zero_cal import ContactGroup, JointZeroCalibrator, KnownDistance
from point_zero_cal.geometric import GeometricCalibrator, GeometricParameter


class PositionBackend:
    dof = 2
    joint_names = ("reference", "joint")
    geometric_parameters = (GeometricParameter("joint", "x", 1),)

    def __init__(self):
        self.positions = np.zeros((4, 3))

    def position_and_jacobian(self, q):
        return np.array([q[0], q[1], 0.0]), np.zeros((3, 2))

    def position_and_geometric_jacobian(self, q, parameters):
        position, _ = self.position_and_jacobian(q)
        position[0] += parameters[0]
        return position, np.zeros((3, 1))

    def position_batch(self, q):
        self.positions[:, :2] = q
        return self.positions

    def geometric_position_batch(self, q, parameters):
        positions = self.position_batch(q)
        positions[:, 0] += parameters[0]
        return positions


def setup_evaluation(geometric):
    backend = PositionBackend()
    groups = [ContactGroup("a", [[0, 1], [0, 2]]), ContactGroup("b", [[3, 1], [3, 2]])]
    distances = [KnownDistance("a", "b", 3)]
    if geometric:
        solver = GeometricCalibrator(backend)
        method = "geometric_position_batch"

        def evaluate():
            return solver.evaluate_groups(groups, [0.3], distances=distances)
    else:
        solver = JointZeroCalibrator(backend)
        method = "position_batch"

        def evaluate():
            return solver.evaluate_groups(
                groups, offsets_rad=[0.3, 0.1], distances=distances
            )

    return backend, solver, method, evaluate


@pytest.mark.parametrize("geometric", [False, True])
def test_position_only_report_matches_scalar_fallback(geometric, monkeypatch):
    backend, solver, method, evaluate = setup_evaluation(geometric)
    with monkeypatch.context() as patch:
        patch.setattr(backend, method, None)
        expected = evaluate().to_dict()

    def forbid_jacobians(*args):
        raise AssertionError("held-out evaluation must not compute Jacobians")

    monkeypatch.setattr(solver, "_evaluate", forbid_jacobians)
    assert evaluate().to_dict() == expected


@pytest.mark.parametrize("geometric", [False, True])
@pytest.mark.parametrize("problem", ["shape", "empty", "nan", "inf", "complex"])
def test_position_only_outputs_are_validated(geometric, problem, monkeypatch):
    backend, _, method, evaluate = setup_evaluation(geometric)
    output = np.zeros((4, 3))
    if problem == "shape":
        output = output[0]
    elif problem == "empty":
        output = output[:0]
    elif problem == "nan":
        output[0, 0] = np.nan
    elif problem == "inf":
        output[0, 0] = np.inf
    else:
        output = output.astype(complex)
        output[0, 0] = 1j
    monkeypatch.setattr(backend, method, lambda *args: output)
    with pytest.raises(ValueError, match="shape|non-finite|real numeric"):
        evaluate()


@pytest.mark.parametrize("geometric", [False, True])
def test_position_only_evaluation_detaches_backend_buffers(geometric):
    backend, solver, _, _ = setup_evaluation(geometric)
    q = np.arange(8, dtype=float).reshape(4, 2)
    if geometric:
        positions = solver._evaluate_positions(q, np.array([0.3]))
        expected = positions.copy()
        solver._evaluate_positions(-q, np.array([-0.4]))
    else:
        positions = solver._evaluate_positions(q)
        expected = positions.copy()
        solver._evaluate_positions(-q)
    np.testing.assert_array_equal(positions, expected)
    assert not np.shares_memory(positions, backend.positions)
