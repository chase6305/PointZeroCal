import numpy as np
import pytest

from point_zero_cal import JointZeroCalibrator


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
