"""Small URDF fixtures shared by backend and CLI tests."""

import pytest

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


@pytest.fixture
def planar_urdf(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF, encoding="utf-8")
    return path
