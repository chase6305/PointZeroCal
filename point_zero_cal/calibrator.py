"""Joint zero-offset calibration from fixed-point contact samples."""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Kinematics(Protocol):
    """Minimal kinematics interface required by the calibrator."""

    @property
    def dof(self) -> int: ...

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def position_and_jacobian(
        self, configuration: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable calibration output with convergence and observability metrics.

    ``initial_point_rms`` and ``corrected_point_rms`` measure the 3D dispersion
    around the sample centroid in metres. ``rank`` and ``singular_values``
    describe only the joints that were not listed in ``fixed_joints``.
    """

    converged: bool
    offsets_rad: NDArray[np.float64]
    iterations: int
    residual_rms: float
    step_norm: float
    rank: int
    condition_number: float
    message: str
    sample_count: int
    active_joint_names: tuple[str, ...]
    singular_values: NDArray[np.float64]
    initial_point_rms: float
    corrected_point_rms: float

    @property
    def offsets_deg(self) -> NDArray[np.float64]:
        """Estimated offsets converted from radians to degrees."""
        return np.rad2deg(self.offsets_rad)


class JointZeroCalibrator:
    """Estimate joint offsets from configurations touching one fixed point.

    Each row in ``samples`` is a complete actuated-joint configuration. At
    least two rows are required, and diverse configurations are essential for
    the offsets to be observable.
    """

    def __init__(
        self,
        kinematics: Kinematics,
        *,
        max_iterations: int = 100,
        step_tolerance: float = 1e-8,
        residual_tolerance: float = 1e-6,
        damping: float = 1e-10,
        rank_tolerance: float = 1e-8,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if (
            step_tolerance <= 0
            or residual_tolerance <= 0
            or damping < 0
            or rank_tolerance <= 0
        ):
            raise ValueError("tolerances must be positive and damping non-negative")
        self.kinematics = kinematics
        self.max_iterations = max_iterations
        self.step_tolerance = step_tolerance
        self.residual_tolerance = residual_tolerance
        self.damping = damping
        self.rank_tolerance = rank_tolerance

    def calibrate(
        self,
        samples: ArrayLike,
        *,
        sample_unit: str = "rad",
        fixed_joints: Sequence[int | str] = (0,),
        initial_offsets: ArrayLike | None = None,
    ) -> CalibrationResult:
        """Estimate offsets that make all sampled TCP positions coincide.

        Args:
            samples: Matrix shaped ``(sample_count, dof)``.
            sample_unit: Either ``"rad"`` or ``"deg"``.
            fixed_joints: Joint indices or URDF names excluded from estimation.
                The first joint is fixed by default because a common base
                rotation is a gauge freedom of fixed-point-only calibration.
            initial_offsets: Optional radian seed shaped ``(dof,)``.

        Returns:
            A :class:`CalibrationResult`; non-convergence is reported in the
            result rather than raised as an exception.

        Raises:
            ValueError: If inputs or backend outputs have invalid shapes or
                non-finite values.
        """
        q = self._validate_samples(samples, sample_unit)
        offsets = self._initial_offsets(initial_offsets)
        active = self._active_indices(fixed_joints)
        if active.size == 0:
            raise ValueError("at least one joint must remain calibratable")

        step_norm = float("inf")
        initial_positions, _ = self._evaluate(q)
        initial_point_rms = self._point_rms(initial_positions)

        for iteration in range(1, self.max_iterations + 1):
            positions, jacobians = self._evaluate(q + offsets)
            # Linearize p_i(q_i + offset + step) around the current offset.
            # Centering removes the unknown fixed contact point and treats all
            # samples symmetrically instead of privileging the first pose.
            matrix, error = self._linear_system(positions, jacobians, active)
            if self.damping:
                augmented_matrix = np.vstack(
                    [matrix, np.sqrt(self.damping) * np.eye(active.size)]
                )
                augmented_error = np.concatenate([error, np.zeros(active.size)])
                step = np.linalg.lstsq(augmented_matrix, augmented_error, rcond=None)[0]
            else:
                step = np.linalg.lstsq(matrix, error, rcond=None)[0]
            offsets[active] += step
            step_norm = float(np.linalg.norm(step, ord=np.inf))
            point_rms = self._point_rms(positions)
            if step_norm <= self.step_tolerance and point_rms <= self.residual_tolerance:
                break

        # Re-evaluate after the final update. Reporting the pre-update residual
        # can falsely characterize a result that stopped at max_iterations.
        final_positions, final_jacobians = self._evaluate(q + offsets)
        final_matrix, _ = self._linear_system(final_positions, final_jacobians, active)
        singular_values = np.linalg.svd(final_matrix, compute_uv=False)
        # A relative threshold represents engineering observability better than
        # NumPy's machine-precision default for nearly zero Jacobian columns.
        cutoff = singular_values[0] * self.rank_tolerance if singular_values.size else 0.0
        rank = int(np.count_nonzero(singular_values > cutoff))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size and singular_values[-1] > cutoff
            else float("inf")
        )
        corrected_point_rms = self._point_rms(final_positions)
        converged = (
            rank == active.size
            and step_norm <= self.step_tolerance
            and corrected_point_rms <= self.residual_tolerance
        )
        if rank < active.size:
            message = f"insufficient observability: rank {rank} < {active.size} active joints"
        elif corrected_point_rms > self.residual_tolerance:
            message = "point residual did not reach tolerance"
        elif step_norm > self.step_tolerance:
            message = "maximum iterations reached before the offset step converged"
        else:
            message = "calibration converged"

        return CalibrationResult(
            converged=converged,
            offsets_rad=offsets.copy(),
            iterations=iteration,
            residual_rms=corrected_point_rms,
            step_norm=step_norm,
            rank=rank,
            condition_number=condition,
            message=message,
            sample_count=len(q),
            active_joint_names=tuple(self.kinematics.joint_names[i] for i in active),
            singular_values=singular_values.copy(),
            initial_point_rms=initial_point_rms,
            corrected_point_rms=corrected_point_rms,
        )

    def _evaluate(
        self, configurations: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate and validate the backend at every configuration."""
        positions = []
        jacobians = []
        for configuration in configurations:
            position, jacobian = self.kinematics.position_and_jacobian(configuration)
            position = np.asarray(position, dtype=float)
            jacobian = np.asarray(jacobian, dtype=float)
            if position.shape != (3,) or jacobian.shape != (3, self.kinematics.dof):
                raise ValueError(
                    "kinematics must return position shape (3,) and "
                    f"jacobian shape (3, {self.kinematics.dof})"
                )
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(jacobian)):
                raise ValueError("kinematics returned non-finite values")
            positions.append(position)
            jacobians.append(jacobian)
        return np.vstack(positions), np.stack(jacobians)

    @staticmethod
    def _linear_system(
        positions: NDArray[np.float64],
        jacobians: NDArray[np.float64],
        active: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build ``(J_i - mean(J)) step = mean(p) - p_i``."""
        centered_positions = positions - positions.mean(axis=0)
        centered_jacobians = jacobians - jacobians.mean(axis=0)
        matrix = centered_jacobians.reshape(-1, jacobians.shape[-1])[:, active]
        return matrix, -centered_positions.reshape(-1)

    @staticmethod
    def _point_rms(positions: NDArray[np.float64]) -> float:
        """Return RMS Euclidean distance from the point centroid."""
        centered = positions - positions.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))

    def _validate_samples(self, samples: ArrayLike, unit: str) -> NDArray[np.float64]:
        q = np.asarray(samples, dtype=float)
        if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != self.kinematics.dof:
            raise ValueError(
                f"samples must have shape (N, {self.kinematics.dof}) with N >= 2"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("samples must contain only finite values")
        if unit == "deg":
            return np.deg2rad(q)
        if unit != "rad":
            raise ValueError("sample_unit must be 'rad' or 'deg'")
        return q.copy()

    def _initial_offsets(self, value: ArrayLike | None) -> NDArray[np.float64]:
        offsets = np.zeros(self.kinematics.dof) if value is None else np.asarray(value, dtype=float)
        if offsets.shape != (self.kinematics.dof,) or not np.all(np.isfinite(offsets)):
            raise ValueError(f"initial_offsets must have shape ({self.kinematics.dof},)")
        return offsets.copy()

    def _active_indices(self, fixed_joints: Sequence[int | str]) -> NDArray[np.int64]:
        names = self.kinematics.joint_names
        fixed: set[int] = set()
        for joint in fixed_joints:
            if isinstance(joint, str):
                if joint not in names:
                    raise ValueError(f"unknown joint name: {joint}")
                fixed.add(names.index(joint))
            elif isinstance(joint, (int, np.integer)) and 0 <= int(joint) < self.kinematics.dof:
                fixed.add(int(joint))
            else:
                raise ValueError(f"invalid fixed joint: {joint!r}")
        return np.asarray([i for i in range(self.kinematics.dof) if i not in fixed], dtype=int)
