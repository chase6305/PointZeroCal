import numpy as np
import pytest

from point_zero_cal import PinocchioKinematics


URDF = """<?xml version="1.0"?>
<robot name="planar">
  <link name="base"/>
  <link name="link1"/>
  <link name="tool"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="tool"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="tip" type="fixed">
    <parent link="tool"/><child link="tip_link"/><origin xyz="1 0 0"/>
  </joint>
  <link name="tip_link"/>
</robot>
"""


def test_pinocchio_jacobian_matches_finite_difference(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF, encoding="utf-8")
    backend = PinocchioKinematics(urdf, "tip_link")
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


def test_offset_tcp_jacobian_matches_finite_difference(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF, encoding="utf-8")
    backend = PinocchioKinematics(urdf, "tool", tcp_xyz=(0.13, -0.04, 0.08))
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


def test_validates_tcp_configuration_and_joint_limits(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF, encoding="utf-8")
    backend = PinocchioKinematics(urdf, "tip_link")

    assert backend.within_joint_limits(np.zeros(2))
    assert not backend.within_joint_limits(np.array([4.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        backend.position_and_jacobian(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="tcp_xyz"):
        PinocchioKinematics(urdf, "tip_link", tcp_xyz=(0.0, np.nan, 0.0))
