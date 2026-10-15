"""Finite-difference validation of the geometric Pinocchio backend."""

from pathlib import Path

import numpy as np
import pytest

from point_zero_cal.geometric import GeometricParameter
from point_zero_cal.geometric_backend import PinocchioGeometricKinematics
from point_zero_cal.pinocchio_backend import PinocchioKinematics

UR10 = Path(__file__).resolve().parents[1] / "assets/UR10/UR10.urdf"

COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")


def parameters(joint_names, components=COMPONENTS):
    return [
        GeometricParameter(joint, component, prior_scale=0.01)
        for joint in joint_names
        for component in components
    ]


def assert_columns_match_finite_differences(backend, configuration, deviations=None):
    """Check every Jacobian column against deviated-evaluation differences."""
    size = len(backend.geometric_parameters)
    if deviations is None:
        deviations = np.zeros(size)
    position, jacobian = backend.position_and_geometric_jacobian(
        configuration, deviations
    )
    epsilon = 1e-7
    for column in range(size):
        plus = deviations.copy()
        minus = deviations.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        plus_position, _ = backend.position_and_geometric_jacobian(configuration, plus)
        minus_position, _ = backend.position_and_geometric_jacobian(
            configuration, minus
        )
        finite = (plus_position - minus_position) / (2 * epsilon)
        np.testing.assert_allclose(
            jacobian[:, column],
            finite,
            atol=5e-8,
            err_msg=backend.geometric_parameters[column].label,
        )
    return position


def test_planar_parameters_match_finite_differences(planar_urdf):
    backend = PinocchioGeometricKinematics(
        planar_urdf,
        "tip_link",
        parameters(("joint2", "tip")),
    )
    assert_columns_match_finite_differences(backend, np.array([0.3, -0.7]))


def test_ur10_all_components_match_finite_differences():
    backend = PinocchioGeometricKinematics(
        UR10,
        "ee_link",
        parameters(("Joint2", "Joint5", "jointEE")),
        tcp_xyz=(0.08, 0, 0.16),
    )
    assert_columns_match_finite_differences(
        backend, np.array([0.1, -1.0, 1.5, -1.2, 0.7, 0.3])
    )


def test_ur10_derivatives_at_nonzero_deviations():
    backend = PinocchioGeometricKinematics(
        UR10,
        "ee_link",
        parameters(("Joint2", "Joint5", "jointEE")),
        tcp_xyz=(0.08, 0, 0.16),
    )
    assert_columns_match_finite_differences(
        backend,
        np.array([0.1, -1.0, 1.5, -1.2, 0.7, 0.3]),
        np.linspace(-0.2, 0.2, len(backend.geometric_parameters)),
    )


def test_parent_link_frame_formula_through_an_interleaved_fixed_joint(tmp_path):
    """A fixed joint between revolute joints must use the parent link frame."""
    urdf = tmp_path / "chain.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="chain">
  <link name="base"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="tool"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="mid" type="fixed">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.2 0 0" rpy="0 0 0.5"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link2"/><child link="tool"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    backend = PinocchioGeometricKinematics(
        urdf, "tool", parameters(("joint2", "mid")), tcp_xyz=(0.5, 0, 0)
    )
    assert_columns_match_finite_differences(backend, np.array([0.4, -0.9]))
    deviations = np.linspace(-0.05, 0.05, len(backend.geometric_parameters))
    assert_columns_match_finite_differences(backend, np.array([0.4, -0.9]), deviations)
    # Check the physical semantics against a separately rebuilt model.
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf)
    for name in ("joint2", "mid"):
        origin = tree.getroot().find(f"joint[@name='{name}']/origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        for parameter, value in zip(
            backend.geometric_parameters, deviations, strict=True
        ):
            if parameter.joint_name == name:
                index = COMPONENTS.index(parameter.component)
                (xyz if index < 3 else rpy)[index % 3] += value
        origin.set("xyz", " ".join(map(str, xyz)))
        origin.set("rpy", " ".join(map(str, rpy)))
    rebuilt_path = tmp_path / "rebuilt.urdf"
    tree.write(rebuilt_path)
    rebuilt = PinocchioKinematics(rebuilt_path, "tool", tcp_xyz=(0.5, 0, 0))
    q = np.array([0.4, -0.9])
    np.testing.assert_allclose(
        backend.position_and_geometric_jacobian(q, deviations)[0],
        rebuilt.tcp_pose(q).translation,
        atol=1e-12,
    )


@pytest.mark.parametrize("first_translation", ["0.2 0 0", "0 0 0"])
def test_chained_fixed_joint_is_rejected(tmp_path, first_translation):
    urdf = tmp_path / "chained.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="chained">
  <link name="base"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="tool"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="mid1" type="fixed">
    <parent link="link1"/><child link="link2"/><origin xyz="0.2 0 0"/>
  </joint>
  <joint name="mid2" type="fixed">
    <parent link="link2"/><child link="tool"/><origin xyz="0.3 0 0"/>
  </joint>
</robot>
""".replace('xyz="0.2 0 0"', f'xyz="{first_translation}"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="chained fixed transform"):
        PinocchioGeometricKinematics(urdf, "tool", parameters(("mid2",)))
    # The first fixed joint of the chain is single-level and stays supported.
    backend = PinocchioGeometricKinematics(urdf, "tool", parameters(("mid1",)))
    assert backend.geometric_parameters[0].joint_name == "mid1"
    assert_columns_match_finite_differences(backend, np.array([0.4]))


def test_parameters_outside_tcp_ancestry_have_zero_derivatives(planar_urdf):
    backend = PinocchioGeometricKinematics(
        planar_urdf, "link1", parameters(("joint2", "tip")), tcp_xyz=(0.1, 0.2, 0.3)
    )
    q = np.array([0.4, -0.9])
    deviations = np.linspace(-0.05, 0.05, len(backend.geometric_parameters))
    assert_columns_match_finite_differences(backend, q, deviations)
    np.testing.assert_array_equal(
        backend.position_and_geometric_jacobian(q, deviations)[1], 0.0
    )


def test_combined_rotation_columns_preserve_parameter_order_and_ignore_descendants():
    selected = [
        GeometricParameter(joint, component, 0.01)
        for joint, component in (
            ("Joint5", "yaw"),
            ("Joint2", "pitch"),
            ("Joint1", "z"),
            ("jointEE", "roll"),
            ("Joint3", "roll"),
            ("Joint2", "x"),
            ("Joint1", "yaw"),
        )
    ]
    backend = PinocchioGeometricKinematics(
        UR10, "Link3", selected, tcp_xyz=(0.08, 0, 0.16)
    )
    q = np.array([0.1, -1.0, 1.5, -1.2, 0.7, 0.3])
    deviations = np.linspace(-0.05, 0.05, len(selected))
    assert_columns_match_finite_differences(backend, q, deviations)
    jacobian = backend.position_and_geometric_jacobian(q, deviations)[1]
    np.testing.assert_array_equal(jacobian[:, [0, 3]], 0.0)


def test_nominal_model_is_restored_after_deviated_evaluations():
    backend = PinocchioGeometricKinematics(
        UR10, "ee_link", parameters(("Joint2", "jointEE")), tcp_xyz=(0.08, 0, 0.16)
    )
    q = np.array([0.1, -1.0, 1.5, -1.2, 0.7, 0.3])
    position, jacobian = backend.position_and_jacobian(q)
    deviations = np.linspace(-0.01, 0.01, len(backend.geometric_parameters))
    backend.position_and_geometric_jacobian(q, deviations)
    restored_position, restored_jacobian = backend.position_and_jacobian(q)
    np.testing.assert_array_equal(restored_position, position)
    np.testing.assert_array_equal(restored_jacobian, jacobian)
    np.testing.assert_allclose(backend.tcp_pose(q).translation, position, atol=1e-12)


def test_deviated_positions_match_a_rebuilt_perturbed_urdf(tmp_path):
    """The deviation semantics must equal rewriting the URDF origins."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(UR10)
    perturb = {
        "Joint2": {"y": 0.003, "pitch": -0.02},
        "jointEE": {"z": 0.004},
        "Joint5": {"roll": 0.015},
    }
    for joint in tree.getroot().findall("joint"):
        name = joint.get("name")
        if name not in perturb:
            continue
        origin = joint.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
        rpy = np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()])
        for component, value in perturb[name].items():
            if component in ("x", "y", "z"):
                xyz["xyz".index(component)] += value
            else:
                rpy[["roll", "pitch", "yaw"].index(component)] += value
        origin.set("xyz", " ".join(repr(float(v)) for v in xyz))
        origin.set("rpy", " ".join(repr(float(v)) for v in rpy))
    path = tmp_path / "perturbed.urdf"
    tree.write(path)
    parameters = [
        GeometricParameter("Joint2", "y", 0.01),
        GeometricParameter("Joint2", "pitch", 0.01),
        GeometricParameter("jointEE", "z", 0.01),
        GeometricParameter("Joint5", "roll", 0.01),
    ]
    deviations = np.array([0.003, -0.02, 0.004, 0.015])
    backend = PinocchioGeometricKinematics(
        UR10, "ee_link", parameters, tcp_xyz=(0.08, 0, 0.16)
    )
    rebuilt = PinocchioKinematics(path, "ee_link", tcp_xyz=(0.08, 0, 0.16))
    q = np.array([0.1, -1.0, 1.5, -1.2, 0.7, 0.3])
    deviated, _ = backend.position_and_geometric_jacobian(q, deviations)
    rebuilt_position = rebuilt.position_and_jacobian(q)[0]
    np.testing.assert_allclose(deviated, rebuilt_position, atol=1e-12)


def test_backend_rejects_invalid_parameter_sets(planar_urdf):
    with pytest.raises(ValueError, match="GeometricParameter"):
        PinocchioGeometricKinematics(planar_urdf, "tip_link", ["joint1.x"])
    with pytest.raises(ValueError, match="at least one"):
        PinocchioGeometricKinematics(planar_urdf, "tip_link", [])
    with pytest.raises(ValueError, match="unknown joint"):
        PinocchioGeometricKinematics(planar_urdf, "tip_link", parameters(("missing",)))
    with pytest.raises(ValueError, match="unique"):
        PinocchioGeometricKinematics(
            planar_urdf,
            "tip_link",
            [GeometricParameter("joint1", "x", 0.01)] * 2,
        )
    backend = PinocchioGeometricKinematics(
        planar_urdf, "tip_link", parameters(("joint1",))
    )
    q = np.array([0.2, 0.3])
    size = len(backend.geometric_parameters)
    with pytest.raises(ValueError, match="shape"):
        backend.position_and_geometric_jacobian(q, np.zeros(size + 1))
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros(size)
        bad[0] = np.nan
        backend.position_and_geometric_jacobian(q, bad)


def test_inherited_nominal_kinematics_remain_available(planar_urdf):
    backend = PinocchioGeometricKinematics(
        planar_urdf, "tip_link", parameters(("joint2",))
    )
    q = np.array([0.2, -0.4])
    position, jacobian = backend.position_and_jacobian(q)
    assert position.shape == (3,)
    assert jacobian.shape == (3, 2)
    assert backend.dof == 2
    assert backend.joint_names == ("joint1", "joint2")
    assert backend.within_joint_limits(q)
    assert [p.label for p in backend.geometric_parameters] == [
        "joint2.x",
        "joint2.y",
        "joint2.z",
        "joint2.roll",
        "joint2.pitch",
        "joint2.yaw",
    ]


def test_model_restored_when_forward_kinematics_raises(planar_urdf, monkeypatch):
    backend = PinocchioGeometricKinematics(
        planar_urdf, "tip_link", parameters(("joint2", "tip"))
    )
    q = np.array([0.2, -0.4])
    expected = backend.tcp_pose(q).homogeneous.copy()

    def fail(*args):
        raise RuntimeError("injected FK failure")

    with monkeypatch.context() as patch:
        patch.setattr(backend._pin, "forwardKinematics", fail)
        with pytest.raises(RuntimeError, match="injected FK failure"):
            backend.position_and_geometric_jacobian(q, np.full(12, 0.01))
    np.testing.assert_array_equal(backend.tcp_pose(q).homogeneous, expected)


@pytest.mark.parametrize("robot", ["planar", "ur10"])
@pytest.mark.parametrize("deviation", [0.0, 0.05])
def test_batch_evaluation_matches_scalar_and_restores_model(
    planar_urdf, robot, deviation
):
    rng = np.random.default_rng(106)
    if robot == "planar":
        backend = PinocchioGeometricKinematics(
            planar_urdf,
            "tip_link",
            parameters(("joint2", "tip")),
            tcp_xyz=(0.08, 0, 0.16),
        )
    else:
        backend = PinocchioGeometricKinematics(
            UR10,
            "ee_link",
            parameters(("Joint2", "Joint5", "jointEE")),
            tcp_xyz=(0.08, 0, 0.16),
        )
    configurations = rng.uniform(-1, 1, size=(10, backend.dof))
    deviations = rng.uniform(-deviation, deviation, len(backend.geometric_parameters))
    configurations_before = configurations.copy()
    deviations_before = deviations.copy()
    nominal = backend.position_and_jacobian(configurations[0])
    scalar = [
        backend.position_and_geometric_jacobian(q, deviations) for q in configurations
    ]
    positions, jacobians = backend.position_and_geometric_jacobian_batch(
        configurations, deviations
    )
    np.testing.assert_array_equal(positions, np.array([s[0] for s in scalar]))
    np.testing.assert_array_equal(jacobians, np.array([s[1] for s in scalar]))
    np.testing.assert_array_equal(configurations, configurations_before)
    np.testing.assert_array_equal(deviations, deviations_before)
    restored = backend.position_and_jacobian(configurations[0])
    for original, actual in zip(nominal, restored, strict=True):
        np.testing.assert_array_equal(actual, original)
    positions_before, jacobians_before = positions.copy(), jacobians.copy()
    backend.position_and_geometric_jacobian_batch(configurations, -deviations)
    np.testing.assert_array_equal(positions, positions_before)
    np.testing.assert_array_equal(jacobians, jacobians_before)


@pytest.mark.parametrize(
    "configurations",
    [
        [0.1, 0.2],
        np.zeros((0, 2)),
        np.zeros((2, 3)),
        [[0, np.nan]],
        [[0, np.inf]],
        [[0, 1j]],
    ],
)
@pytest.mark.parametrize(
    "method", ["position_and_geometric_jacobian_batch", "geometric_position_batch"]
)
def test_batch_rejects_invalid_configurations(planar_urdf, configurations, method):
    backend = PinocchioGeometricKinematics(
        planar_urdf, "tip_link", parameters(("tip",))
    )
    with pytest.raises(ValueError, match="configurations"):
        getattr(backend, method)(configurations, np.zeros(6))


@pytest.mark.parametrize("deviations", [np.zeros(5), [np.nan] * 6, [1j] * 6])
@pytest.mark.parametrize(
    "method", ["position_and_geometric_jacobian_batch", "geometric_position_batch"]
)
def test_batch_rejects_invalid_parameters(planar_urdf, deviations, method):
    backend = PinocchioGeometricKinematics(
        planar_urdf, "tip_link", parameters(("tip",))
    )
    with pytest.raises(ValueError, match="parameters"):
        getattr(backend, method)([[0.1, 0.2]], deviations)


@pytest.mark.parametrize(
    "method", ["position_and_geometric_jacobian_batch", "geometric_position_batch"]
)
def test_batch_restores_model_after_partial_fk_failure(
    planar_urdf, monkeypatch, method
):
    backend = PinocchioGeometricKinematics(
        planar_urdf, "tip_link", parameters(("joint2", "tip"))
    )
    q = np.array([0.2, -0.4])
    expected = backend.position_and_jacobian(q)
    forward = backend._pin.forwardKinematics
    calls = 0

    def fail_on_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected partial batch failure")
        return forward(*args)

    with monkeypatch.context() as patch:
        patch.setattr(backend._pin, "forwardKinematics", fail_on_second)
        with pytest.raises(RuntimeError, match="partial batch failure"):
            getattr(backend, method)(np.array([q, -q, q]), np.full(12, 0.01))
    assert calls == 2
    for original, restored in zip(
        expected, backend.position_and_jacobian(q), strict=True
    ):
        np.testing.assert_array_equal(restored, original)


@pytest.mark.parametrize("components", [COMPONENTS, ("x", "y", "z")])
def test_geometric_positions_match_full_batch_without_derivatives(
    planar_urdf, monkeypatch, components
):
    backend = PinocchioGeometricKinematics(
        planar_urdf,
        "tool",
        parameters(("joint2", "tip"), components),
        tcp_xyz=(0.13, -0.04, 0.08),
        tcp_rpy=(0.1, -0.2, 0.3),
    )
    q = np.array([[0.3, -0.7], [-0.8, 0.5], [0.1, 0.2]])
    deviations = np.linspace(-0.1, 0.1, len(backend.geometric_parameters))
    nominal = backend.position_batch(q)
    expected, _ = backend.position_and_geometric_jacobian_batch(q, deviations)

    def forbid_derivatives(*args):
        raise AssertionError("position evaluation must not compute derivatives")

    monkeypatch.setattr(backend, "_rpy_tangent_axes", forbid_derivatives)
    monkeypatch.setattr(backend._pin, "computeJointJacobians", forbid_derivatives)
    actual = backend.geometric_position_batch(q, deviations)
    backend.geometric_position_batch(q, -deviations)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(backend.position_batch(q), nominal)
