"""Batch kinematics preserve calibration results and backend validation."""

import numpy as np
import pytest

from point_zero_cal import (
    ContactGroup,
    JointZeroCalibrator,
    KnownDistance,
    PinocchioKinematics,
)


@pytest.mark.parametrize("grouped", [False, True])
def test_joint_batch_and_scalar_produce_identical_fits(
    planar_urdf, grouped, monkeypatch
):
    backend = PinocchioKinematics(planar_urdf, "tip_link")
    solver = JointZeroCalibrator(backend)
    offsets = np.array([0.0, -0.025])
    first = np.array([[0, np.pi / 2], [np.pi / 2, -np.pi / 2]]) - offsets
    second = np.array([[0, -np.pi / 2], [-np.pi / 2, np.pi / 2]]) - offsets
    if grouped:
        groups = [
            ContactGroup("a", first, sigma_m=0.001),
            ContactGroup("b", second, sigma_m=0.002),
        ]
        distances = [KnownDistance("a", "b", 2, sigma_m=0.003)]
        batch = solver.calibrate_groups(groups, distances=distances)
        monkeypatch.setattr(backend, "position_and_jacobian_batch", None)
        scalar = solver.calibrate_groups(groups, distances=distances)
    else:
        batch = solver.calibrate(first)
        monkeypatch.setattr(backend, "position_and_jacobian_batch", None)
        scalar = solver.calibrate(first)
    assert batch.converged
    assert batch.to_dict() == scalar.to_dict()
    np.testing.assert_allclose(batch.offsets_rad, offsets, atol=1e-7)


class BatchBackend:
    dof = 2
    joint_names = ("reference", "joint")

    def __init__(self):
        self.positions = np.zeros((2, 3))
        self.jacobians = np.zeros((2, 3, 2))

    def position_and_jacobian_batch(self, q):
        self.positions[:, :2] = q
        self.jacobians[:, 0, 0] = q[:, 0]
        return self.positions, self.jacobians

    def position_and_jacobian(self, q):
        raise AssertionError("a batch backend must use its batch method")


@pytest.mark.parametrize("problem", ["positions", "jacobians", "nan", "inf", "complex"])
def test_joint_batch_outputs_are_validated(problem, monkeypatch):
    backend = BatchBackend()
    positions, jacobians = np.zeros((2, 3)), np.zeros((2, 3, 2))
    if problem == "positions":
        positions = positions[0]
    elif problem == "jacobians":
        jacobians = jacobians[..., 0]
    elif problem == "nan":
        positions[0, 0] = np.nan
    elif problem == "inf":
        jacobians[0, 0, 0] = np.inf
    else:
        positions = positions.astype(complex)
        positions[0, 0] = 1j
    monkeypatch.setattr(
        backend, "position_and_jacobian_batch", lambda q: (positions, jacobians)
    )
    with pytest.raises(ValueError, match="shape|non-finite|real numeric"):
        JointZeroCalibrator(backend)._evaluate(np.zeros((2, 2)))


def test_joint_calibrator_detaches_batch_buffers():
    solver = JointZeroCalibrator(BatchBackend())
    q = np.array([[0.3, -0.7], [-0.8, 0.5]])
    positions, jacobians = solver._evaluate(q)
    saved_positions, saved_jacobians = positions.copy(), jacobians.copy()
    solver._evaluate(-q)
    np.testing.assert_array_equal(positions, saved_positions)
    np.testing.assert_array_equal(jacobians, saved_jacobians)


def test_nonfinite_corrected_configuration_fails_before_batch_call():
    with pytest.raises(ValueError, match="corrected configurations"):
        JointZeroCalibrator(BatchBackend())._evaluate(np.array([[np.inf, 0], [0, 0]]))
