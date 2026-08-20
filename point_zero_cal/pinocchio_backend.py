"""Pinocchio kinematics adapter."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PinocchioKinematics:
    """Pinocchio-backed kinematics for a fixed-base, single-DoF-joint robot.

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
        tcp_translation = np.asarray(tcp_xyz, dtype=float)
        tcp_rotation = np.asarray(tcp_rpy, dtype=float)
        if (
            tcp_translation.shape != (3,)
            or tcp_rotation.shape != (3,)
            or not np.all(np.isfinite(tcp_translation))
            or not np.all(np.isfinite(tcp_rotation))
        ):
            raise ValueError("tcp_xyz and tcp_rpy must each contain three finite values")
        self.tcp_placement = pin.SE3(
            pin.rpy.rpyToMatrix(tcp_rotation), tcp_translation
        )
        actuated_joint_ids = tuple(range(1, self.model.njoints))
        if any(self.model.joints[joint_id].nv != 1 for joint_id in actuated_joint_ids):
            raise ValueError("multi-DoF joints are not supported")
        self._joint_names = tuple(self.model.names[joint_id] for joint_id in actuated_joint_ids)

    @property
    def dof(self) -> int:
        return int(self.model.nv)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def position_and_jacobian(
        self, configuration: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        q = self._validate_configuration(configuration)
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        frame_pose = self.data.oMf[self.frame_id]
        tcp_offset_world = frame_pose.rotation @ self.tcp_placement.translation
        position = frame_pose.translation + tcp_offset_world
        # A point offset r from the frame origin has velocity
        # v_tcp = v_frame + omega x r. This term is what makes an eccentric
        # TCP sensitive to rotation about the final wrist joint.
        angular = np.asarray(jacobian[3:], dtype=float)
        point_jacobian = np.asarray(jacobian[:3], dtype=float) + np.cross(
            angular.T, tcp_offset_world
        ).T
        return np.asarray(position, dtype=float).copy(), point_jacobian.copy()

    def tcp_pose(self, configuration: NDArray[np.float64]) -> Any:
        """Return the world-to-TCP pose for simulation and diagnostics."""
        q = self._validate_configuration(configuration)
        self._pin.framesForwardKinematics(self.model, self.data, q)
        return self._pin.SE3(self.data.oMf[self.frame_id] * self.tcp_placement)

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
        q = np.asarray(configuration, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"configuration must have shape ({self.dof},)")
        if not np.all(np.isfinite(q)):
            raise ValueError("configuration must contain only finite values")
        return q
