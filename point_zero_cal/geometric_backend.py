"""Pinocchio backend with TCP derivatives with respect to URDF origins.

Each parameter targets one component of a joint's ``<origin>`` transform.
Translation columns are the parent link's world rotation; rotation columns
use the infinitesimal axis of the URDF rpy convention at the joint's current
deviated rpy. Formulas are validated against rebuilt-perturbed-URDF finite
differences in the test suite.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._urdf import parse_joints as _parse_urdf_joints
from ._validation import real_array
from .geometric import GeometricParameter
from .pinocchio_backend import PinocchioKinematics

_COMPONENT_AXIS = {
    "x": 0,
    "y": 1,
    "z": 2,
    "roll": 0,
    "pitch": 1,
    "yaw": 2,
}
_TRANSLATION_COMPONENTS = ("x", "y", "z")


class PinocchioGeometricKinematics(PinocchioKinematics):
    """Pinocchio kinematics with an explicit geometric parameter model.

    Parameters may target revolute model joints or single-level fixed joints
    (a fixed joint whose own URDF origin is carried by its child link's body
    frame placement). Chained fixed joints are rejected: a folded frame
    placement is no longer equivalent to perturbing one URDF origin. The
    nominal model is restored after every call, so zero-offset fits on the
    same object stay exact.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        end_frame: str,
        parameters: Sequence[GeometricParameter],
        *,
        tcp_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tcp_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        path = Path(urdf_path).expanduser()
        urdf_joints = _parse_urdf_joints(path)
        super().__init__(path, end_frame, tcp_xyz=tcp_xyz, tcp_rpy=tcp_rpy)
        pin = self._pin
        parent_joint = {info[2]: name for name, info in urdf_joints.items()}
        self._parameters = tuple(parameters)
        if not self._parameters:
            raise ValueError("parameters must contain at least one GeometricParameter")
        if any(
            not isinstance(parameter, GeometricParameter)
            for parameter in self._parameters
        ):
            raise ValueError("parameters must contain GeometricParameter values")
        labels = [parameter.label for parameter in self._parameters]
        if len(set(labels)) != len(labels):
            raise ValueError("geometric parameter labels must be unique")
        self._parameter_info: list[tuple[str, int, int]] = []
        self._joint_parameters: dict[str, list[tuple[int, str]]] = {}
        self._joint_origin: dict[str, tuple[NDArray, NDArray]] = {}
        self._origin_placements: dict[str, object] = {}
        for index, parameter in enumerate(self._parameters):
            if parameter.joint_name not in urdf_joints:
                raise ValueError(f"unknown joint name: {parameter.joint_name}")
            joint_type, parent_link, child_link, rpy, xyz = urdf_joints[parameter.joint_name]
            parent_name = parent_joint.get(parent_link)
            if (
                joint_type == "fixed"
                and parent_name is not None
                and urdf_joints[parent_name][0] == "fixed"
            ):
                raise ValueError(
                    f"joint {parameter.joint_name!r} is part of a chained fixed transform; "
                    "parameters must target single-level fixed joints"
                )
            if not parent_link or not child_link:
                raise ValueError(
                    f"joint {parameter.joint_name!r} must declare parent and child links"
                )
            if not self.model.existFrame(parent_link):
                raise ValueError(
                    f"parent link {parent_link!r} of joint "
                    f"{parameter.joint_name!r} not found in the model"
                )
            kind, entity_id = self._resolve_entity(
                parameter.joint_name, child_link, rpy, xyz, pin
            )
            self._parameter_info.append(
                (kind, entity_id, self.model.getFrameId(parent_link))
            )
            self._joint_parameters.setdefault(parameter.joint_name, []).append(
                (index, parameter.component)
            )
            if parameter.joint_name not in self._joint_origin:
                self._joint_origin[parameter.joint_name] = (rpy.copy(), xyz.copy())
                self._origin_placements[parameter.joint_name] = pin.SE3(
                    pin.rpy.rpyToMatrix(rpy), xyz
                )

        # URDF ancestry, rather than placement equality, determines which
        # parameters affect the TCP and which folded entities must move.
        def ancestors(link: str) -> list[str]:
            result = []
            while link in parent_joint:
                name = parent_joint[link]
                result.append(name)
                link = urdf_joints[name][1]
            return result

        end_link = urdf_joints[end_frame][2] if end_frame in urdf_joints else end_frame
        tcp_ancestors = set(ancestors(end_link))
        self._affects_tcp = np.array(
            [parameter.joint_name in tcp_ancestors for parameter in self._parameters]
        )
        self._joint_parameters = dict(
            sorted(
                self._joint_parameters.items(),
                key=lambda item: len(ancestors(urdf_joints[item[0]][2])),
            )
        )
        self._derivative_columns = {}
        for name, entries in self._joint_parameters.items():
            columns = []
            for translation in (True, False):
                selected = [
                    (index, _COMPONENT_AXIS[component])
                    for index, component in entries
                    if (component in _TRANSLATION_COMPONENTS) == translation
                ]
                columns.append((
                    np.array([index for index, _ in selected], dtype=int),
                    np.array([axis for _, axis in selected], dtype=int),
                ))
            self._derivative_columns[name] = tuple(columns)
        self._folded_entities = {}
        for name, entries in self._joint_parameters.items():
            kind, entity_id, _ = self._parameter_info[entries[0][0]]
            if kind != "frame":
                continue
            parent_id = self.model.frames[entity_id].parentJoint
            joint_ids = tuple(
                joint_id
                for joint_id in range(1, self.model.njoints)
                if self.model.parents[joint_id] == parent_id
                and name in ancestors(urdf_joints[self.model.names[joint_id]][2])
            )
            frame_ids = []
            for frame_id, frame in enumerate(self.model.frames):
                link = (
                    urdf_joints[frame.name][2]
                    if frame.name in urdf_joints else frame.name
                )
                if frame.parentJoint == parent_id and name in ancestors(link):
                    frame_ids.append(frame_id)
            self._folded_entities[name] = joint_ids, tuple(frame_ids)

    def _resolve_entity(
        self,
        name: str,
        child_link: str,
        rpy: NDArray[np.float64],
        xyz: NDArray[np.float64],
        pin: object,
    ) -> tuple[str, int]:
        """Map a URDF joint to a model joint or a fold-free child-link frame.

        Fixed joints fold their origin into the child link's body frame, so
        that frame is the entity whose placement carries the parameter. The
        entity is accepted only when its placement equals the joint's own
        origin; chained fixed transforms fold several origins into one
        placement and are rejected.
        """
        nominal = pin.SE3(pin.rpy.rpyToMatrix(rpy), xyz)
        if name in self.joint_names:
            return "joint", self.model.getJointId(name)
        if not self.model.existFrame(child_link):
            raise ValueError(f"joint {name!r} not found in the model")
        frame_id = self.model.getFrameId(child_link)
        if self.model.frames[frame_id].type != pin.BODY:
            raise ValueError(f"joint {name!r} is not a supported model entity")
        if not np.allclose(
            self.model.frames[frame_id].placement.homogeneous,
            nominal.homogeneous,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(
                f"joint {name!r} is part of a chained fixed transform; "
                "parameters must target single-level fixed joints"
            )
        return "frame", frame_id

    @staticmethod
    def _rpy_tangent_axes(
        rpy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Parent-frame rotation axes for URDF Rz(yaw) Ry(pitch) Rx(roll)."""
        _, pitch, yaw = rpy
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([[cy * cp, -sy, 0], [sy * cp, cy, 0], [-sp, 0, 1]])

    @property
    def geometric_parameters(self) -> tuple[GeometricParameter, ...]:
        return self._parameters

    @property
    def parameter_labels(self) -> tuple[str, ...]:
        """Parameter labels in geometric Jacobian column order."""
        return tuple(parameter.label for parameter in self._parameters)

    def position_and_geometric_jacobian(
        self,
        configuration: NDArray[np.float64],
        parameters: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return TCP position (3,) and Jacobian (3, n_parameters) at the
        nonlinear model whose URDF origins are deviated by ``parameters``.

        Model placements are temporarily deviated and restored, so this call
        never leaks state into later evaluations.
        """
        q = self._validate_configuration(configuration)
        values = real_array(parameters, "parameters")
        if values.shape != (len(self._parameters),) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"parameters must have shape ({len(self._parameters)},) with finite values"
            )
        pin = self._pin
        saved_joints = {}
        saved_frames = {}
        tangent_axes = {}
        try:
            for name, entries in self._joint_parameters.items():
                kind, entity_id, _ = self._parameter_info[entries[0][0]]
                if kind == "joint":
                    # Copy explicitly: vector element access may alias the slot.
                    saved_joints[entity_id] = pin.SE3(
                        self.model.jointPlacements[entity_id]
                    )
                else:
                    joint_ids, frame_ids = self._folded_entities[name]
                    for joint_id in joint_ids:
                        saved_joints[joint_id] = pin.SE3(
                            self.model.jointPlacements[joint_id]
                        )
                    for frame_id in frame_ids:
                        saved_frames[frame_id] = pin.SE3(
                            self.model.frames[frame_id].placement
                        )
            for name, entries in self._joint_parameters.items():
                kind, entity_id, _ = self._parameter_info[entries[0][0]]
                rpy, xyz = self._joint_origin[name]
                dr = np.zeros(3)
                dt = np.zeros(3)
                for index, component in entries:
                    axis = _COMPONENT_AXIS[component]
                    if component in _TRANSLATION_COMPONENTS:
                        dt[axis] = values[index]
                    else:
                        dr[axis] = values[index]
                deviated = pin.SE3(pin.rpy.rpyToMatrix(rpy + dr), xyz + dt)
                tangent_axes[name] = self._rpy_tangent_axes(rpy + dr)
                # The pinocchio entity placement becomes placement * D where
                # D = M(0)^{-1} * M(x), exactly matching an exported URDF.
                delta = self._origin_placements[name].actInv(deviated)
                if kind == "joint":
                    self.model.jointPlacements[entity_id] = (
                        self.model.jointPlacements[entity_id] * delta
                    )
                else:
                    # Fixed origins are folded into descendant placements.
                    # Move all entities stored relative to the same movable
                    # parent; ordinary FK propagates through movable children.
                    current = self.model.frames[entity_id].placement
                    correction = current * delta * current.inverse()
                    joint_ids, frame_ids = self._folded_entities[name]
                    for joint_id in joint_ids:
                        self.model.jointPlacements[joint_id] = (
                            correction * self.model.jointPlacements[joint_id]
                        )
                    for frame_id in frame_ids:
                        self.model.frames[frame_id].placement = (
                            correction * self.model.frames[frame_id].placement
                        )
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pose = self.data.oMf[self.frame_id] * self.tcp_placement
            position = np.asarray(pose.translation, dtype=float).copy()
            jacobian = np.empty((3, len(self._parameters)))
            for name, entries in self._joint_parameters.items():
                index = entries[0][0]
                kind, entity_id, parent_frame_id = self._parameter_info[index]
                translation, rotation = self._derivative_columns[name]
                translation_indices, translation_axes = translation
                rotation_indices, rotation_axes = rotation
                if not self._affects_tcp[index]:
                    jacobian[:, translation_indices] = 0.0
                    jacobian[:, rotation_indices] = 0.0
                    continue
                if kind == "joint":
                    anchor = self.data.oMi[entity_id].translation
                else:
                    anchor = self.data.oMf[entity_id].translation
                parent_rotation = self.data.oMf[parent_frame_id].rotation
                jacobian[:, translation_indices] = parent_rotation[:, translation_axes]
                if rotation_indices.size:
                    world_axes = parent_rotation @ tangent_axes[name][:, rotation_axes]
                    jacobian[:, rotation_indices] = np.cross(
                        world_axes.T, position - anchor
                    ).T
            return position, jacobian
        finally:
            for entity_id, placement in saved_joints.items():
                self.model.jointPlacements[entity_id] = placement
            for entity_id, placement in saved_frames.items():
                self.model.frames[entity_id].placement = placement
