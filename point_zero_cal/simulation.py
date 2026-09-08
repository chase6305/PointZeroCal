"""Synthetic fixed-point contact data generation with Pinocchio IK."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite_scalar, positive_integer, real_array
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
    max_angle_deg = finite_scalar(max_angle_deg, "max_angle_deg")
    levels = positive_integer(levels, "levels")
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
    count = positive_integer(count, "count")
    if count < 3 or count % 2 == 0:
        raise ValueError("count must be an odd integer >= 3")
    max_angle_deg = finite_scalar(max_angle_deg, "max_angle_deg")

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
    target_xyz: ArrayLike | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
    damping: float = 1e-6,
) -> FixedPointDataset:
    """Solve poses with constant TCP XYZ and varied world-frame RPY deltas.

    The solver uses a damped resolved-rate update in a world-aligned frame.
    Each target starts from the same seed to favour the same IK branch.
    Generated configurations must satisfy the URDF position limits.
    ``target_xyz`` optionally selects a different fixed contact point in metres;
    the default is the TCP position at the seed configuration.
    """
    pin = kinematics._pin
    seed = real_array(seed_configuration, "seed_configuration")
    if seed.shape != (kinematics.dof,):
        raise ValueError(f"seed_configuration must have shape ({kinematics.dof},)")
    if not np.all(np.isfinite(seed)):
        raise ValueError("seed_configuration must contain only finite values")
    if not kinematics.within_joint_limits(seed):
        raise ValueError("seed_configuration is outside URDF joint limits")
    max_iterations = positive_integer(max_iterations, "max_iterations")
    tolerance = finite_scalar(tolerance, "tolerance")
    damping = finite_scalar(damping, "damping")
    if isinstance(rpy_deltas, np.ndarray):
        deltas = real_array(rpy_deltas, "rpy_deltas")
    else:
        try:
            rows = tuple(tuple(delta) for delta in rpy_deltas)
        except TypeError as exc:
            raise ValueError(
                "rpy_deltas must be an iterable of three-value rows"
            ) from exc
        deltas = real_array(rows, "rpy_deltas")
    if (
        deltas.ndim != 2
        or deltas.shape[0] < 1
        or deltas.shape[1] != 3
        or not np.all(np.isfinite(deltas))
    ):
        raise ValueError(
            "rpy_deltas must have shape (N, 3), N >= 1, with finite values"
        )

    nominal_pose = kinematics.tcp_pose(seed)
    target_xyz = real_array(
        nominal_pose.translation if target_xyz is None else target_xyz, "target_xyz"
    ).copy()
    if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
        raise ValueError("target_xyz must have shape (3,) with finite metre values")
    solutions: list[NDArray[np.float64]] = []
    errors: list[float] = []
    regularizer = damping * np.eye(6)

    for delta_rpy in deltas:
        target_rotation = pin.rpy.rpyToMatrix(delta_rpy) @ nominal_pose.rotation
        q = seed.copy()
        # Evaluate the final update as well: a solution reached on the last
        # allowed step must not be reported as an IK failure.
        for iteration in range(max_iterations + 1):
            current, jacobian = kinematics.tcp_pose_and_jacobian(q)
            error = np.concatenate(
                [
                    target_xyz - current.translation,
                    pin.log3(target_rotation @ current.rotation.T),
                ]
            )
            if float(np.linalg.norm(error)) <= tolerance:
                break
            if iteration == max_iterations:
                raise RuntimeError(
                    f"IK failed for RPY delta {delta_rpy.tolist()}; "
                    f"final error={np.linalg.norm(error):.3e}"
                )
            # Damped least squares remains stable near wrist singularities.
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + regularizer, error
            )
            q = pin.integrate(kinematics.model, q, 0.5 * step)
        if not kinematics.within_joint_limits(q):
            raise RuntimeError(
                f"IK solution violates joint limits for {delta_rpy.tolist()}"
            )
        solutions.append(np.asarray(q).copy())
        errors.append(float(np.linalg.norm(error[:3])))

    return FixedPointDataset(
        configurations_rad=np.vstack(solutions),
        target_xyz=target_xyz,
        rpy_deltas_rad=deltas,
        position_errors=np.asarray(errors),
    )
