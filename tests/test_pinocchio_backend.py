import numpy as np
import pytest

from point_zero_cal import PinocchioKinematics


def test_pinocchio_jacobian_matches_finite_difference(planar_urdf):
    backend = PinocchioKinematics(planar_urdf, "tip_link")
    q = np.array([0.3, -0.7])

    position, jacobian = backend.position_and_jacobian(q)
    epsilon = 1e-7
    numerical = np.column_stack(
        [
            (backend.position_and_jacobian(q + epsilon * np.eye(2)[i])[0] - position)
            / epsilon
            for i in range(2)
        ]
    )

    assert backend.joint_names == ("joint1", "joint2")
    np.testing.assert_allclose(jacobian, numerical, atol=1e-6)


def test_offset_tcp_jacobian_matches_finite_difference(planar_urdf):
    backend = PinocchioKinematics(planar_urdf, "tool", tcp_xyz=(0.13, -0.04, 0.08))
    q = np.array([-0.2, 0.6])
    position, jacobian = backend.position_and_jacobian(q)
    epsilon = 1e-7
    numerical = np.column_stack(
        [
            (backend.position_and_jacobian(q + epsilon * np.eye(2)[i])[0] - position)
            / epsilon
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(jacobian, numerical, atol=1e-6)


def test_validates_tcp_configuration_and_joint_limits(planar_urdf):
    backend = PinocchioKinematics(planar_urdf, "tip_link")

    assert backend.within_joint_limits(np.zeros(2))
    assert not backend.within_joint_limits(np.array([4.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        backend.position_and_jacobian(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="tcp_xyz"):
        PinocchioKinematics(planar_urdf, "tip_link", tcp_xyz=(0.0, np.nan, 0.0))


def test_tcp_pose_jacobian_and_outputs_survive_later_calls(planar_urdf):
    backend = PinocchioKinematics(
        planar_urdf, "tool", tcp_xyz=(0.13, -0.04, 0.08), tcp_rpy=(0.1, -0.2, 0.3)
    )
    q = np.array([-0.2, 0.6])
    pose, jacobian = backend.tcp_pose_and_jacobian(q)
    expected_pose = pose.homogeneous.copy()
    expected_jacobian = jacobian.copy()
    epsilon = 1e-7
    numerical = np.empty((6, 2))
    for index in range(2):
        perturbed = backend.tcp_pose(q + epsilon * np.eye(2)[index])
        numerical[:3, index] = (perturbed.translation - pose.translation) / epsilon
        numerical[3:, index] = (
            backend._pin.log3(perturbed.rotation @ pose.rotation.T) / epsilon
        )

    np.testing.assert_allclose(jacobian, numerical, atol=1e-6)
    np.testing.assert_array_equal(pose.homogeneous, expected_pose)
    np.testing.assert_array_equal(jacobian, expected_jacobian)
    np.testing.assert_allclose(
        backend.tcp_pose(q).homogeneous, expected_pose, atol=1e-14
    )


def test_prismatic_joints_cannot_be_misinterpreted_as_angles(planar_urdf):
    planar_urdf.write_text(
        planar_urdf.read_text().replace('type="revolute"', 'type="prismatic"')
    )
    with pytest.raises(ValueError, match="prismatic"):
        PinocchioKinematics(planar_urdf, "tip_link")


def test_backend_rejects_complex_configurations_and_tcp_transforms(planar_urdf):
    backend = PinocchioKinematics(planar_urdf, "tip_link")
    with pytest.raises(ValueError, match="configuration.*real"):
        backend.tcp_pose(np.array([0, 1j]))
    for field in ("tcp_xyz", "tcp_rpy"):
        with pytest.raises(ValueError, match=f"{field}.*real"):
            PinocchioKinematics(
                planar_urdf, "tip_link", **{field: np.array([0, 0, 1j])}
            )
