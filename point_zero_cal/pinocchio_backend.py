"""Pinocchio kinematics adapter."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._validation import real_array


class PinocchioKinematics:
    """Pinocchio kinematics for a fixed-base robot with bounded revolute joints.

    ``tcp_xyz`` and ``tcp_rpy`` describe a rigid transform from ``end_frame``
    to the actual contact point. Translations are metres and rotations radians.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        end_frame: str,
        *,
        tcp_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tcp_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError(
                "Pinocchio is required; install this project with `pip install -e .`"
            ) from exc

        path = Path(urdf_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"URDF does not exist: {path}")
        self._pin: Any = pin
        self.model = pin.buildModelFromUrdf(str(path))
        self.data = self.model.createData()
        if not self.model.existFrame(end_frame):
            raise ValueError(f"frame {end_frame!r} not found in URDF")
        if self.model.nq != self.model.nv:
            raise ValueError(
                "only fixed-base models with one configuration variable per velocity are supported"
            )
        self.frame_id = self.model.getFrameId(end_frame)
        tcp_translation = real_array(tcp_xyz, "tcp_xyz")
        tcp_rotation = real_array(tcp_rpy, "tcp_rpy")
        if (
            tcp_translation.shape != (3,)
            or tcp_rotation.shape != (3,)
            or not np.all(np.isfinite(tcp_translation))
            or not np.all(np.isfinite(tcp_rotation))
        ):
            raise ValueError(
                "tcp_xyz and tcp_rpy must each contain three finite values"
            )
        self.tcp_placement = pin.SE3(pin.rpy.rpyToMatrix(tcp_rotation), tcp_translation)
        # The hint selects a path; skipping work still checks the live transform.
        self._check_zero_tcp = not np.any(tcp_translation)
        actuated_joint_ids = tuple(range(1, self.model.njoints))
        if any(self.model.joints[joint_id].nv != 1 for joint_id in actuated_joint_ids):
            raise ValueError("multi-DoF joints are not supported")
        if not actuated_joint_ids:
            raise ValueError("URDF must contain at least one actuated joint")
        if any(
            self.model.joints[joint_id].shortname().startswith("JointModelP")
            for joint_id in actuated_joint_ids
        ):
            raise ValueError("prismatic joints are not supported; offsets use radians")
        self._joint_names = tuple(
            self.model.names[joint_id] for joint_id in actuated_joint_ids
        )

    @property
    def dof(self) -> int:
        return int(self.model.nv)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def position_and_jacobian(
        self, configuration: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return TCP position (3,) and translational Jacobian (3, dof)."""
        pose, jacobian = self.tcp_pose_and_jacobian(configuration)
        return np.asarray(pose.translation, dtype=float).copy(), jacobian[:3]

    def tcp_pose_and_jacobian(
        self, configuration: NDArray[np.float64]
    ) -> tuple[Any, NDArray[np.float64]]:
        """Return TCP pose and world-aligned Jacobian (6, dof).

        Rows 0–2 are linear velocity in m/s; rows 3–5 are angular velocity
        in rad/s. Both outputs own their data across subsequent backend calls.
        """
        return self._tcp_pose_and_jacobian(self._validate_configuration(configuration))

    def _tcp_pose_and_jacobian(
        self, q: NDArray[np.float64]
    ) -> tuple[Any, NDArray[np.float64]]:
        """Evaluate a configuration whose shape and finite values are validated."""
        pin = self._pin
        # This computes joint placements and Jacobians in one traversal. Only
        # the requested frame needs updating, even for a large URDF tree.
        pin.computeJointJacobians(self.model, self.data, q)
        frame_pose = pin.updateFramePlacement(self.model, self.data, self.frame_id)
        jacobian = np.array(
            pin.getFrameJacobian(
                self.model,
                self.data,
                self.frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ),
            dtype=float,
            copy=True,
        )
        # A point offset r from the frame origin has velocity
        # v_tcp = v_frame + omega x r. This term is what makes an eccentric
        # TCP sensitive to rotation about the final wrist joint.
        translation = self.tcp_placement.translation
        if (
            not self._check_zero_tcp
            or translation[0]
            or translation[1]
            or translation[2]
        ):
            tcp_offset_world = frame_pose.rotation @ translation
            jacobian[:3] -= pin.skew(tcp_offset_world) @ jacobian[3:]
        return frame_pose * self.tcp_placement, jacobian

    def position_and_jacobian_batch(
        self, configurations: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return owned positions (N, 3) and Jacobians (N, 3, dof).

        Configurations are radians, positions metres, derivatives m/rad.
        Validate the batch once and preserve scalar overrides in subclasses.
        """
        q = self._validate_configurations(configurations)
        positions = np.empty((len(q), 3))
        jacobians = np.empty((len(q), 3, self.dof))
        standard = (
            getattr(self.position_and_jacobian, "__func__", None)
            is PinocchioKinematics.position_and_jacobian
            and getattr(self.tcp_pose_and_jacobian, "__func__", None)
            is PinocchioKinematics.tcp_pose_and_jacobian
        )
        for index, configuration in enumerate(q):
            if standard:
                pose, jacobian = self._tcp_pose_and_jacobian(configuration)
                positions[index], jacobians[index] = pose.translation, jacobian[:3]
            else:
                position, jacobian = self.position_and_jacobian(configuration)
                position = real_array(position, "kinematics position")
                jacobian = real_array(jacobian, "kinematics jacobian")
                if position.shape != (3,) or jacobian.shape != (3, self.dof):
                    raise ValueError(
                        "kinematics must return position shape (3,) and "
                        f"jacobian shape (3, {self.dof})"
                    )
                if not np.all(np.isfinite(position)) or not np.all(
                    np.isfinite(jacobian)
                ):
                    raise ValueError("kinematics returned non-finite values")
                positions[index], jacobians[index] = position, jacobian
        return positions, jacobians

    def tcp_pose(self, configuration: NDArray[np.float64]) -> Any:
        """Return the world-to-TCP pose for simulation and diagnostics."""
        q = self._validate_configuration(configuration)
        self._pin.forwardKinematics(self.model, self.data, q)
        frame_pose = self._pin.updateFramePlacement(
            self.model, self.data, self.frame_id
        )
        return frame_pose * self.tcp_placement

    def position_batch(
        self, configurations: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return TCP positions (N, 3) in metres without computing Jacobians.

        Configurations have shape (N, dof) in radians, N >= 1. The returned
        array owns its data across subsequent backend calls.
        """
        q = self._validate_configurations(configurations)
        positions = np.empty((len(q), 3))
        for index, configuration in enumerate(q):
            positions[index] = self.tcp_pose(configuration).translation
        return positions

    @property
    def lower_position_limits(self) -> NDArray[np.float64]:
        """URDF lower joint limits in model order."""
        return np.asarray(self.model.lowerPositionLimit, dtype=float).copy()

    @property
    def upper_position_limits(self) -> NDArray[np.float64]:
        """URDF upper joint limits in model order."""
        return np.asarray(self.model.upperPositionLimit, dtype=float).copy()

    def within_joint_limits(self, configuration: NDArray[np.float64]) -> bool:
        """Return whether a finite configuration lies within URDF limits."""
        q = self._validate_configuration(configuration)
        return bool(
            np.all(q >= self.lower_position_limits)
            and np.all(q <= self.upper_position_limits)
        )

    def _validate_configuration(
        self, configuration: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Normalize a configuration array and reject invalid values."""
        q = real_array(configuration, "configuration")
        if q.shape != (self.dof,):
            raise ValueError(f"configuration must have shape ({self.dof},)")
        if not np.all(np.isfinite(q)):
            raise ValueError("configuration must contain only finite values")
        return q

    def _validate_configurations(
        self, configurations: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Normalize a nonempty batch of finite configurations in radians."""
        q = real_array(configurations, "configurations")
        if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != self.dof:
            raise ValueError(f"configurations must have shape (N, {self.dof}), N >= 1")
        if not np.all(np.isfinite(q)):
            raise ValueError("configurations must contain only finite values")
        return q
