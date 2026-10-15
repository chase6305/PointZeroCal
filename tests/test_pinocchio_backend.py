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


@pytest.mark.parametrize("rpy", [(0, 0, 0), (0.2, -0.1, 0.3)])
@pytest.mark.parametrize("initial_xyz", [(0, 0, 0), (0.02, -0.01, 0.03)])
def test_zero_tcp_fast_path_tracks_later_translation_changes(
    planar_urdf, rpy, initial_xyz
):
    backend = PinocchioKinematics(
        planar_urdf, "tip_link", tcp_xyz=initial_xyz, tcp_rpy=rpy
    )
    q = np.array([[0.3, -0.7], [-0.8, 0.5]])
    initial = backend.position_and_jacobian_batch(q)
    for translation in [(0, 0.04, 0.13), (0, 0, 0), initial_xyz]:
        backend.tcp_placement.translation[:] = translation
        actual = backend.position_and_jacobian_batch(q)
        expected = PinocchioKinematics(
            planar_urdf, "tip_link", tcp_xyz=translation, tcp_rpy=rpy
        ).position_and_jacobian_batch(q)
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, reference)
    for actual, expected in zip(
        backend.position_and_jacobian_batch(q), initial, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


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


def test_position_batch_matches_tcp_positions_without_jacobians(
    planar_urdf, monkeypatch
):
    backend = PinocchioKinematics(
        planar_urdf, "tool", tcp_xyz=(0.13, -0.04, 0.08), tcp_rpy=(0.1, -0.2, 0.3)
    )
    q = np.array([[0.3, -0.7], [-0.8, 0.5], [0.1, 0.2]])
    expected = np.array([backend.position_and_jacobian(row)[0] for row in q])

    def forbid_jacobians(*args):
        raise AssertionError("position evaluation must not compute Jacobians")

    monkeypatch.setattr(backend._pin, "computeJointJacobians", forbid_jacobians)
    actual = backend.position_batch(q)
    backend.position_batch(-q)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "configurations",
    [
        [0, 0],
        np.zeros((0, 2)),
        np.zeros((2, 3)),
        [[0, np.nan]],
        [[0, np.inf]],
        [[0, 1j]],
    ],
)
def test_position_batch_rejects_invalid_configurations(planar_urdf, configurations):
    backend = PinocchioKinematics(planar_urdf, "tip_link")
    with pytest.raises(ValueError, match="configurations"):
        backend.position_batch(configurations)


def test_joint_jacobian_batch_matches_scalar_and_owns_outputs(planar_urdf):
    backend = PinocchioKinematics(
        planar_urdf, "tool", tcp_xyz=(0.13, -0.04, 0.08), tcp_rpy=(0.1, -0.2, 0.3)
    )
    q = np.array([[0.3, -0.7], [-0.8, 0.5], [0.1, 0.2]])
    expected = [backend.position_and_jacobian(row) for row in q]
    positions, jacobians = backend.position_and_jacobian_batch(q)
    backend.position_and_jacobian_batch(-q)
    np.testing.assert_array_equal(positions, [value[0] for value in expected])
    np.testing.assert_array_equal(jacobians, [value[1] for value in expected])
    np.testing.assert_array_equal(q, [[0.3, -0.7], [-0.8, 0.5], [0.1, 0.2]])


@pytest.mark.parametrize(
    "configurations",
    [
        [0, 0],
        np.zeros((0, 2)),
        np.zeros((2, 3)),
        [[0, np.nan]],
        [[0, np.inf]],
        [[0, 1j]],
    ],
)
def test_joint_batch_rejects_invalid_configurations(planar_urdf, configurations):
    backend = PinocchioKinematics(planar_urdf, "tip_link")
    with pytest.raises(ValueError, match="configurations"):
        backend.position_and_jacobian_batch(configurations)


def test_joint_batch_preserves_scalar_position_override(planar_urdf):
    class Backend(PinocchioKinematics):
        def position_and_jacobian(self, q):
            position, jacobian = super().position_and_jacobian(q)
            return position + [0.05, -0.01, 0.03], jacobian * 2

    backend = Backend(planar_urdf, "tip_link")
    q = np.array([[0.3, -0.7], [-0.8, 0.5]])
    expected = [backend.position_and_jacobian(row) for row in q]
    actual = backend.position_and_jacobian_batch(q)
    for index in range(2):
        np.testing.assert_array_equal(
            actual[index], [value[index] for value in expected]
        )


def test_joint_batch_preserves_scalar_pose_override(planar_urdf):
    class Backend(PinocchioKinematics):
        def tcp_pose_and_jacobian(self, q):
            pose, jacobian = super().tcp_pose_and_jacobian(q)
            pose.translation += [0.05, -0.01, 0.03]
            return pose, jacobian * 2

    backend = Backend(planar_urdf, "tip_link")
    q = np.array([[0.3, -0.7], [-0.8, 0.5]])
    expected = [backend.position_and_jacobian(row) for row in q]
    actual = backend.position_and_jacobian_batch(q)
    for index in range(2):
        np.testing.assert_array_equal(
            actual[index], [value[index] for value in expected]
        )


@pytest.mark.parametrize(
    "problem", ["position_shape", "jacobian_shape", "nan", "complex"]
)
def test_joint_batch_rejects_invalid_scalar_override_before_next_call(
    planar_urdf, problem
):
    class Backend(PinocchioKinematics):
        calls = 0

        def position_and_jacobian(self, q):
            self.calls += 1
            position, jacobian = super().position_and_jacobian(q)
            if problem == "position_shape":
                position = position[:1]
            elif problem == "jacobian_shape":
                jacobian = jacobian[:, :1]
            elif problem == "nan":
                position[0] = np.nan
            else:
                jacobian = jacobian.astype(complex)
                jacobian[0, 0] = 1j
            return position, jacobian

    backend = Backend(planar_urdf, "tip_link")
    with pytest.raises(ValueError, match="shape|non-finite|real numeric"):
        backend.position_and_jacobian_batch([[0.3, -0.7], [-0.8, 0.5]])
    assert backend.calls == 1
