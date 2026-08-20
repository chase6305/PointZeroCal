"""Synthetic fixed-point contact data generation with Pinocchio IK."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pinocchio_backend import PinocchioKinematics


@dataclass(frozen=True)
class FixedPointDataset:
    """IK-generated configurations sharing a common TCP position."""

    configurations_rad: NDArray[np.float64]
    target_xyz: NDArray[np.float64]
    rpy_deltas_rad: NDArray[np.float64]
    position_errors: NDArray[np.float64]


def make_rpy_grid(
    *, max_angle_deg: float = 12.0, levels: int = 5
) -> NDArray[np.float64]:
    """Create a dense, symmetric roll/pitch/yaw perturbation grid.

    With the defaults, each axis takes five values from -12 to +12 degrees,
    yielding 125 deterministic target orientations.
    """
    if max_angle_deg <= 0:
        raise ValueError("max_angle_deg must be positive")
    if levels < 3 or levels % 2 == 0:
        raise ValueError("levels must be an odd integer >= 3")
    axis = np.deg2rad(np.linspace(-max_angle_deg, max_angle_deg, levels))
    return np.asarray(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T


def make_rpy_samples(
    *, count: int = 41, max_angle_deg: float = 60.0
) -> NDArray[np.float64]:
    """Create an odd-sized, symmetric, low-discrepancy RPY sample set.

    The set contains the zero orientation and positive/negative pairs. The
    default 41 poses span up to 60 degrees per RPY axis, giving broad excitation
    without the 125-pose cost of a full five-level Cartesian grid.
    """
    if count < 3 or count % 2 == 0:
        raise ValueError("count must be an odd integer >= 3")
    if max_angle_deg <= 0:
        raise ValueError("max_angle_deg must be positive")

    def radical_inverse(index: int, base: int) -> float:
        value = 0.0
        factor = 1.0 / base
        while index:
            index, digit = divmod(index, base)
            value += digit * factor
            factor /= base
        return value

    positive = []
    for index in range(1, (count - 1) // 2 + 1):
        unit = np.array([radical_inverse(index, base) for base in (2, 3, 5)])
        positive.append((2.0 * unit - 1.0) * np.deg2rad(max_angle_deg))
    samples = [np.zeros(3)]
    for value in positive:
        samples.extend((value, -value))
    return np.asarray(samples)


def generate_fixed_point_dataset(
    kinematics: PinocchioKinematics,
    seed_configuration: ArrayLike,
    rpy_deltas: Iterable[Iterable[float]],
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
    damping: float = 1e-6,
) -> FixedPointDataset:
    """Solve poses with constant TCP XYZ and varied world-frame RPY deltas.

    The solver uses a damped resolved-rate update in a world-aligned frame.
    Each target starts from the same seed to stay on a continuous IK branch.
    Generated configurations must satisfy the URDF position limits.
    """
    pin = kinematics._pin
    seed = np.asarray(seed_configuration, dtype=float)
    if seed.shape != (kinematics.dof,):
        raise ValueError(f"seed_configuration must have shape ({kinematics.dof},)")
    if not np.all(np.isfinite(seed)):
        raise ValueError("seed_configuration must contain only finite values")
    if not kinematics.within_joint_limits(seed):
        raise ValueError("seed_configuration is outside URDF joint limits")
    if max_iterations < 1 or tolerance <= 0 or damping <= 0:
        raise ValueError("max_iterations, tolerance, and damping must be positive")
    deltas = np.asarray(tuple(rpy_deltas), dtype=float)
    if (
        deltas.ndim != 2
        or deltas.shape[0] < 1
        or deltas.shape[1] != 3
        or not np.all(np.isfinite(deltas))
    ):
        raise ValueError("rpy_deltas must have shape (N, 3), N >= 1, with finite values")

    nominal_pose = kinematics.tcp_pose(seed)
    target_xyz = np.asarray(nominal_pose.translation, dtype=float).copy()
    solutions: list[NDArray[np.float64]] = []
    errors: list[float] = []

    for delta_rpy in deltas:
        target_rotation = pin.rpy.rpyToMatrix(delta_rpy) @ nominal_pose.rotation
        q = seed.copy()
        for _ in range(max_iterations):
            current = kinematics.tcp_pose(q)
            error = np.concatenate(
                [target_xyz - current.translation, pin.log3(target_rotation @ current.rotation.T)]
            )
            if float(np.linalg.norm(error)) <= tolerance:
                break
            frame_jacobian = pin.computeFrameJacobian(
                kinematics.model,
                kinematics.data,
                q,
                kinematics.frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            # TCP translation is expressed in the end frame.
            frame_pose = kinematics.data.oMf[kinematics.frame_id]
            offset_world = frame_pose.rotation @ kinematics.tcp_placement.translation
            linear = np.asarray(frame_jacobian[:3]) + np.cross(
                np.asarray(frame_jacobian[3:]).T, offset_world
            ).T
            jacobian = np.vstack([linear, np.asarray(frame_jacobian[3:])])
            # Damped least squares remains stable near wrist singularities.
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            q = pin.integrate(kinematics.model, q, 0.5 * step)
        else:
            raise RuntimeError(
                f"IK failed for RPY delta {delta_rpy.tolist()}; final error={np.linalg.norm(error):.3e}"
            )
        if not kinematics.within_joint_limits(q):
            raise RuntimeError(f"IK solution violates joint limits for {delta_rpy.tolist()}")
        solutions.append(np.asarray(q).copy())
        errors.append(float(np.linalg.norm(kinematics.tcp_pose(q).translation - target_xyz)))

    return FixedPointDataset(
        configurations_rad=np.vstack(solutions),
        target_xyz=target_xyz,
        rpy_deltas_rad=deltas,
        position_errors=np.asarray(errors),
    )
