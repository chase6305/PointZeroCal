"""Joint zero-offset calibration from fixed-point contact samples."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _groups
from ._least_squares import backtracking_solve, damped_svd_step
from ._validation import finite_scalar, kinematics_metadata, positive_integer, real_array
from .contacts import (
    ContactEvaluation,
    ContactGroup,
    KnownDistance,
)
from .sampling import CandidateRanking, CandidateScore, ObservabilityReport


class Kinematics(Protocol):
    """Minimal kinematics interface required by the calibrator."""

    @property
    def dof(self) -> int: ...

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def position_and_jacobian(
        self, configuration: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return position (3,) in metres and Jacobian (3, dof) in m/rad.

        The configuration is in radians. Returned arrays may reuse backend
        buffers; the calibrator copies their values before the next call.
        """
        ...


@dataclass(frozen=True)
class SampleDiagnostics:
    """Contact-point estimates and per-sample distances in metres.

    Error arrays preserve input row order. ``worst_sample_index`` is zero-based
    and identifies the largest corrected distance, not a proven outlier.
    """

    contact_point_xyz: NDArray[np.float64]
    initial_point_errors: NDArray[np.float64]
    corrected_point_errors: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "contact_point_xyz",
            "initial_point_errors",
            "corrected_point_errors",
        ):
            values = real_array(getattr(self, name), name).copy()
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.contact_point_xyz.shape != (3,):
            raise ValueError("contact_point_xyz must have shape (3,)")
        if (
            self.initial_point_errors.ndim != 1
            or self.initial_point_errors.size < 2
            or self.corrected_point_errors.shape != self.initial_point_errors.shape
            or np.any(self.initial_point_errors < 0)
            or np.any(self.corrected_point_errors < 0)
        ):
            raise ValueError(
                "point errors must be non-negative arrays shaped (N,), N >= 2"
            )

    @property
    def max_corrected_point_error(self) -> float:
        """Largest corrected Euclidean distance to the fitted contact point."""
        return float(np.max(self.corrected_point_errors))

    @property
    def worst_sample_index(self) -> int:
        """Zero-based index of the first sample with the largest residual."""
        return int(np.argmax(self.corrected_point_errors))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostics with explicit metre units."""
        return {
            "contact_point_xyz_m": self.contact_point_xyz.tolist(),
            "initial_point_errors_m": self.initial_point_errors.tolist(),
            "corrected_point_errors_m": self.corrected_point_errors.tolist(),
            "max_corrected_point_error_m": self.max_corrected_point_error,
            "worst_sample_index": self.worst_sample_index,
        }


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable calibration output with convergence and observability metrics.

    ``initial_point_rms`` and ``corrected_point_rms`` measure the 3D dispersion
    around the sample centroid in metres. ``rank`` and ``singular_values``
    describe only the joints that were not listed in ``fixed_joints``.
    ``step_norm`` estimates the remaining undamped offset correction in
    radians at the returned offsets, rather than the last accepted step.
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
    sample_diagnostics: SampleDiagnostics | None = None
    initial_evaluation: ContactEvaluation | None = None
    evaluation: ContactEvaluation | None = None

    def __post_init__(self) -> None:
        for name in ("offsets_rad", "singular_values"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def offsets_deg(self) -> NDArray[np.float64]:
        """Estimated offsets converted from radians to degrees."""
        return np.rad2deg(self.offsets_rad)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible output, using null for infinite condition numbers."""
        payload: dict[str, object] = {
            "converged": self.converged,
            "message": self.message,
            "sample_count": self.sample_count,
            "active_joint_names": list(self.active_joint_names),
            "offsets_rad": self.offsets_rad.tolist(),
            "offsets_deg": self.offsets_deg.tolist(),
            "iterations": self.iterations,
            "residual_rms": self.residual_rms,
            "initial_point_rms": self.initial_point_rms,
            "corrected_point_rms": self.corrected_point_rms,
            "step_norm": self.step_norm,
            "rank": self.rank,
            "condition_number": (
                self.condition_number if np.isfinite(self.condition_number) else None
            ),
            "singular_values": self.singular_values.tolist(),
        }
        if self.sample_diagnostics is not None:
            payload["sample_diagnostics"] = self.sample_diagnostics.to_dict()
        if self.initial_evaluation is not None:
            payload["initial_evaluation"] = self.initial_evaluation.to_dict()
        if self.evaluation is not None:
            payload["evaluation"] = self.evaluation.to_dict()
        return payload


@dataclass
class _FitState:
    positions: NDArray[np.float64]
    jacobians: NDArray[np.float64]
    matrix: NDArray[np.float64]
    error: NDArray[np.float64]
    score: float


@dataclass
class _FitResult:
    offsets: NDArray[np.float64]
    initial: _FitState
    current: _FitState
    iterations: int
    step_norm: float
    singular_values: NDArray[np.float64]
    stalled: bool


_GroupPlan = _groups.GroupPlan


class JointZeroCalibrator:
    """Estimate joint offsets from configurations touching one fixed point.

    Each row in ``samples`` is a complete actuated-joint configuration. At
    least two rows are required, and diverse configurations are essential for
    the offsets to be observable. Damped SVD steps use ``rank_tolerance`` as
    a relative singular-value cutoff; backtracking accepts only reductions in
    point RMS. Step tolerance is in radians and residual tolerance in metres.
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
        self.kinematics = kinematics
        kinematics_metadata(kinematics.dof, kinematics.joint_names)
        self.max_iterations = positive_integer(max_iterations, "max_iterations")
        self.step_tolerance = finite_scalar(step_tolerance, "step_tolerance")
        self.residual_tolerance = finite_scalar(
            residual_tolerance, "residual_tolerance"
        )
        self.damping = finite_scalar(damping, "damping", allow_zero=True)
        self.rank_tolerance = finite_scalar(rank_tolerance, "rank_tolerance")
        if self.rank_tolerance >= 1:
            raise ValueError("rank_tolerance must be less than 1")

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
            initial_offsets: Optional radian seed shaped ``(dof,)``. Fixed
                joints retain their seed values. Apply results by adding
                ``offsets_rad`` to measured configurations in radians.

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

        fit = self._solve_offsets(q, offsets, active, self._single_state)
        offsets = fit.offsets
        positions = fit.current.positions
        point_rms = fit.current.score
        initial_point_rms = fit.initial.score
        initial_point_errors = self._point_errors(fit.initial.positions)
        step_norm, singular_values = fit.step_norm, fit.singular_values
        stalled, iteration = fit.stalled, fit.iterations

        # The cached SVD and remaining correction describe the returned offsets,
        # including when the final allowed update reaches the solution.
        # A relative threshold represents engineering observability better than
        # NumPy's machine-precision default for nearly zero Jacobian columns.
        cutoff = (
            singular_values[0] * self.rank_tolerance if singular_values.size else 0.0
        )
        rank = int(np.count_nonzero(singular_values > cutoff))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if rank == active.size
            else float("inf")
        )
        corrected_point_rms = point_rms
        converged = (
            not stalled
            and rank == active.size
            and step_norm <= self.step_tolerance
            and corrected_point_rms <= self.residual_tolerance
        )
        if rank < active.size:
            message = (
                f"insufficient observability: rank {rank} < {active.size} active joints"
            )
        elif corrected_point_rms > self.residual_tolerance:
            message = "point residual did not reach tolerance"
        elif stalled:
            message = "no residual-decreasing offset step found"
        elif step_norm > self.step_tolerance:
            message = "maximum iterations reached before the offset step converged"
        else:
            message = "calibration converged"

        return CalibrationResult(
            converged=converged,
            offsets_rad=offsets,
            iterations=iteration,
            residual_rms=corrected_point_rms,
            step_norm=step_norm,
            rank=rank,
            condition_number=condition,
            message=message,
            sample_count=len(q),
            active_joint_names=tuple(self.kinematics.joint_names[i] for i in active),
            singular_values=singular_values,
            initial_point_rms=initial_point_rms,
            corrected_point_rms=corrected_point_rms,
            sample_diagnostics=SampleDiagnostics(
                contact_point_xyz=positions[0] - self._center(positions)[0],
                initial_point_errors=initial_point_errors,
                corrected_point_errors=self._point_errors(positions),
            ),
        )

    def calibrate_groups(
        self,
        groups: Sequence[ContactGroup],
        *,
        distances: Sequence[KnownDistance] = (),
        sample_unit: str = "rad",
        fixed_joints: Sequence[int | str] = (0,),
        initial_offsets: ArrayLike | None = None,
        distance_tolerance: float | None = None,
    ) -> CalibrationResult:
        """Fit shared radian offsets to separately centred contact groups.

        The squared objective is the mean of group point-RMS squares plus
        ``sum(distance.weight * distance_error_m**2)``. Each group therefore
        has equal weight regardless of its sample count. A single group without
        distances and without ``sigma_m`` uses the original single-point solve.

        Convergence requires full active rank, a small remaining correction,
        every group's RMS <= ``residual_tolerance``, and every unweighted
        distance error <= ``distance_tolerance`` (defaults to residual tolerance).
        The result's ``initial_evaluation`` and ``evaluation`` contain per-group
        diagnostics; ``sample_diagnostics`` is reserved for the single-point API.
        """
        q, plan = self._prepare_groups(groups, distances, sample_unit)
        offsets = self._initial_offsets(initial_offsets)
        active = self._active_indices(fixed_joints)
        if not active.size:
            raise ValueError("at least one joint must remain calibratable")
        distance_tolerance = (
            self.residual_tolerance
            if distance_tolerance is None
            else finite_scalar(distance_tolerance, "distance_tolerance")
        )

        def evaluate(configurations, active):
            return self._group_state(configurations, active, plan)

        fit = self._solve_offsets(q, offsets, active, evaluate)
        initial = self._group_report(fit.initial.positions, offsets, plan)
        final = self._group_report(fit.current.positions, fit.offsets, plan)
        values = fit.singular_values
        cutoff = values[0] * self.rank_tolerance if values.size else 0.0
        rank = int(np.count_nonzero(values > cutoff))
        condition = (
            float(values[0] / values[-1]) if rank == active.size else float("inf")
        )
        if rank < active.size:
            message = (
                f"insufficient observability: rank {rank} < {active.size} active joints"
            )
        elif any(group.point_rms_m > self.residual_tolerance for group in final.groups):
            message = "contact group residual did not reach tolerance"
        elif any(
            abs(distance.error_m) > distance_tolerance for distance in final.distances
        ):
            message = "distance residual did not reach tolerance"
        elif fit.stalled:
            message = "no residual-decreasing offset step found"
        elif fit.step_norm > self.step_tolerance:
            message = "maximum iterations reached before the offset step converged"
        else:
            message = "calibration converged"
        return CalibrationResult(
            converged=message == "calibration converged",
            offsets_rad=fit.offsets,
            iterations=fit.iterations,
            residual_rms=final.residual_rms_m,
            step_norm=fit.step_norm,
            rank=rank,
            condition_number=condition,
            message=message,
            sample_count=len(q),
            active_joint_names=tuple(self.kinematics.joint_names[i] for i in active),
            singular_values=values,
            initial_point_rms=initial.point_rms_m,
            corrected_point_rms=final.point_rms_m,
            initial_evaluation=initial,
            evaluation=final,
        )

    def evaluate_groups(
        self,
        groups: Sequence[ContactGroup],
        *,
        offsets_rad: ArrayLike,
        distances: Sequence[KnownDistance] = (),
        sample_unit: str = "rad",
    ) -> ContactEvaluation:
        """Evaluate fixed offsets on held-out contact groups, without fitting.

        Group centroids are inferred from these samples; the report measures
        contact consistency and relative distances, not absolute world accuracy.
        No training state is retained. The caller must supply independent groups.
        """
        q, plan = self._prepare_groups(groups, distances, sample_unit)
        offsets = self._initial_offsets(offsets_rad)
        with np.errstate(over="ignore", invalid="ignore"):
            configurations = q + offsets
        positions, _ = self._evaluate(configurations)
        return self._group_report(positions, offsets, plan)

    def rank_contact_candidates(
        self,
        groups: Sequence[ContactGroup],
        candidates: ArrayLike,
        *,
        group_name: str,
        distances: Sequence[KnownDistance] = (),
        sample_unit: str = "rad",
        fixed_joints: Sequence[int | str] = (0,),
        offsets_rad: ArrayLike | None = None,
    ) -> CandidateRanking:
        """Rank feasible next poses by their prospective local observability.

        ``candidates`` has shape ``(candidate_count, dof)`` and uses the same
        unit and measured-joint convention as ``groups``. Each candidate is
        appended hypothetically to ``group_name``; it is not recorded, sent to
        a robot, or used to update offsets. Candidates must already have been
        checked by IK or an operator to keep the ball seated in that group's
        physical socket.

        Rankings prioritize effective rank, then the smallest observable
        singular value, then condition number. The SVD matrix is normalized by
        its row count so one appended candidate can be compared fairly. This is
        a local linearized recommendation around ``offsets_rad`` (zero by
        default), not a guarantee of convergence or collision-free motion.
        """
        q, plan = self._prepare_groups(groups, distances, sample_unit)
        candidate_q = self._validate_candidates(candidates, sample_unit)
        offsets = self._initial_offsets(offsets_rad)
        active = self._active_indices(fixed_joints)
        if not active.size:
            raise ValueError("at least one joint must remain calibratable")
        try:
            target = tuple(group.name for group in plan.groups).index(group_name)
        except ValueError as exc:
            raise ValueError(f"unknown contact group: {group_name!r}") from exc

        with np.errstate(over="ignore", invalid="ignore"):
            configurations = q + offsets
        positions, jacobians = self._evaluate(configurations)
        group_positions = tuple(positions[section] for section in plan.sections)
        group_jacobians = tuple(jacobians[section] for section in plan.sections)
        baseline_matrix = self._contact_information_matrix(
            group_positions, group_jacobians, plan, active
        )
        baseline = self._observability_report(baseline_matrix, active)
        if not plan.distances:
            # Appending to one contact changes only its centred block. All
            # other groups retain their rows with a common sample-count scale.
            section = plan.sections[target]
            start, stop = 3 * section.start, 3 * section.stop
            prospective_matrix = np.empty((len(baseline_matrix) + 3, len(active)))
            count_scale = np.sqrt((len(q) + 1) / len(q))
            prospective_matrix[:start] = baseline_matrix[:start] * count_scale
            prospective_matrix[stop + 3 :] = baseline_matrix[stop:] * count_scale
            target_scale = np.sqrt(
                (len(q) + 1) / (len(plan.groups) * (section.stop - section.start + 1))
            )
            if plan.groups[target].sigma_m is not None:
                target_scale /= plan.groups[target].sigma_m
        scores = []
        for source_index, candidate in enumerate(candidate_q):
            with np.errstate(over="ignore", invalid="ignore"):
                candidate_configuration = candidate + offsets
            candidate_position, candidate_jacobian = self._evaluate(
                candidate_configuration[None, :]
            )
            if not plan.distances:
                augmented = np.concatenate(
                    (group_jacobians[target], candidate_jacobian)
                )
                prospective_matrix[start : stop + 3] = (
                    self._center(augmented[:, :, active]) * target_scale
                ).reshape(-1, len(active))
                if not np.all(np.isfinite(prospective_matrix)):
                    raise ValueError(
                        "weighted contact Jacobian exceeds the floating-point range"
                    )
            else:
                prospective_positions = list(group_positions)
                prospective_jacobians = list(group_jacobians)
                prospective_positions[target] = np.vstack(
                    (prospective_positions[target], candidate_position)
                )
                prospective_jacobians[target] = np.vstack(
                    (prospective_jacobians[target], candidate_jacobian)
                )
                prospective_matrix = self._contact_information_matrix(
                    tuple(prospective_positions),
                    tuple(prospective_jacobians),
                    plan,
                    active,
                )
            report = self._observability_report(prospective_matrix, active)
            scores.append(
                CandidateScore(
                    source_index,
                    candidate,
                    report,
                    report.rank - baseline.rank,
                )
            )
        return CandidateRanking(group_name, baseline, tuple(scores))

    def _prepare_groups(
        self,
        groups: Sequence[ContactGroup],
        distances: Sequence[KnownDistance],
        unit: str,
    ) -> tuple[NDArray[np.float64], _GroupPlan]:
        return _groups.prepare_groups(groups, distances, unit, self.kinematics.dof)

    def _validate_candidates(
        self, candidates: ArrayLike, unit: str
    ) -> NDArray[np.float64]:
        """Validate one or more prospective measured configurations."""
        values = real_array(candidates, "candidates")
        if (
            values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != self.kinematics.dof
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "candidates must have shape (N, "
                f"{self.kinematics.dof}) with N >= 1 and finite values"
            )
        if unit == "deg":
            return np.deg2rad(values)
        if unit == "rad":
            return values.copy()
        raise ValueError("sample_unit must be 'rad' or 'deg'")

    def _contact_information_matrix(
        self,
        group_positions: tuple[NDArray[np.float64], ...],
        group_jacobians: tuple[NDArray[np.float64], ...],
        plan: _GroupPlan,
        active: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Build the grouped residual Jacobian without evaluating residual values."""
        return _groups.contact_information_matrix(
            group_positions, group_jacobians, plan, active
        )

    def _observability_report(
        self, matrix: NDArray[np.float64], active: NDArray[np.int64]
    ) -> ObservabilityReport:
        """Return a row-count-normalized spectrum for comparing candidates."""
        normalized = matrix / np.sqrt(matrix.shape[0])
        values = np.linalg.svd(normalized, compute_uv=False)
        cutoff = values[0] * self.rank_tolerance if values.size else 0.0
        rank = int(np.count_nonzero(values > cutoff))
        fully_observable = rank == len(active)
        condition = float(values[0] / values[-1]) if fully_observable else float("inf")
        smallest = float(values[rank - 1]) if rank else 0.0
        return ObservabilityReport(
            tuple(self.kinematics.joint_names[index] for index in active),
            rank,
            values,
            smallest,
            condition,
            matrix.shape[0],
        )

    def _group_centres(
        self, positions: NDArray[np.float64], plan: _GroupPlan
    ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        return _groups.group_centres(positions, plan)

    def _group_state(
        self,
        configurations: NDArray[np.float64],
        active: NDArray[np.int64],
        plan: _GroupPlan,
    ) -> _FitState:
        if (
            len(plan.groups) == 1
            and not plan.distances
            and plan.groups[0].sigma_m is None
        ):
            return self._single_state(configurations, active)
        positions, jacobians = self._evaluate(configurations)
        matrix, residual = _groups.group_state_system(
            positions, jacobians, plan, active
        )
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(residual)):
            raise ValueError(
                "weighted contact residuals exceed the floating-point range"
            )
        score = float(np.hypot.reduce(residual / np.sqrt(len(positions))))
        if not np.isfinite(score):
            raise ValueError("contact objective exceeds the floating-point range")
        return _FitState(positions, jacobians, matrix, -residual, score)

    def _group_report(
        self,
        positions: NDArray[np.float64],
        offsets: NDArray[np.float64],
        plan: _GroupPlan,
    ) -> ContactEvaluation:
        return _groups.group_report(positions, plan, offsets)

    def _single_state(
        self, configurations: NDArray[np.float64], active: NDArray[np.int64]
    ) -> _FitState:
        positions, jacobians = self._evaluate(configurations)
        matrix, error = self._linear_system(positions, jacobians, active)
        return _FitState(
            positions, jacobians, matrix, error, self._point_rms(positions)
        )

    def _solve_offsets(
        self,
        q: NDArray[np.float64],
        offsets: NDArray[np.float64],
        active: NDArray[np.int64],
        evaluate: Callable[[NDArray[np.float64], NDArray[np.int64]], _FitState],
    ) -> _FitResult:
        """Share cached evaluations, SVD updates and backtracking across objectives."""
        x0 = offsets[active].copy()

        def evaluate_active(x: NDArray[np.float64]) -> _FitState:
            full = offsets.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                full[active] = x
                configurations = q + full
            return evaluate(configurations, active)

        def apply_step(
            x: NDArray[np.float64], step: NDArray[np.float64]
        ) -> NDArray[np.float64] | None:
            full = offsets.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = x + step
                full[active] = candidate
                configurations = q + full
            if not np.all(np.isfinite(full)) or not np.all(np.isfinite(configurations)):
                return None
            return candidate

        fit = backtracking_solve(
            x0,
            evaluate_active,
            apply_step,
            max_iterations=self.max_iterations,
            step_tolerance=self.step_tolerance,
            damping=self.damping,
            rank_tolerance=self.rank_tolerance,
        )
        full = offsets.copy()
        full[active] = fit.parameters
        return _FitResult(
            full,
            fit.initial,
            fit.current,
            fit.iterations,
            fit.step_norm,
            fit.singular_values,
            fit.stalled,
        )

    def _evaluate(
        self, configurations: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate and validate the backend at every configuration."""
        if not np.all(np.isfinite(configurations)):
            raise ValueError("corrected configurations must contain only finite values")
        positions = np.empty((len(configurations), 3))
        jacobians = np.empty((len(configurations), 3, self.kinematics.dof))
        for index, configuration in enumerate(configurations):
            position, jacobian = self.kinematics.position_and_jacobian(configuration)
            position = real_array(position, "kinematics position")
            jacobian = real_array(jacobian, "kinematics jacobian")
            if position.shape != (3,) or jacobian.shape != (3, self.kinematics.dof):
                raise ValueError(
                    "kinematics must return position shape (3,) and "
                    f"jacobian shape (3, {self.kinematics.dof})"
                )
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(jacobian)):
                raise ValueError("kinematics returned non-finite values")
            positions[index] = position
            jacobians[index] = jacobian
        return positions, jacobians

    def _offset_step(
        self, matrix: NDArray[np.float64], error: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
        """Return a damped step, undamped remaining-step norm, and spectrum."""
        return damped_svd_step(
            matrix, error, damping=self.damping, rank_tolerance=self.rank_tolerance
        )

    @staticmethod
    def _center(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Center rows without subtracting a rounded large common baseline."""
        return _groups.center(values)

    @staticmethod
    def _linear_system(
        positions: NDArray[np.float64],
        jacobians: NDArray[np.float64],
        active: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build ``(J_i - mean(J)) step = mean(p) - p_i``."""
        centered_positions = JointZeroCalibrator._center(positions)
        active_jacobians = jacobians[:, :, active]
        centered_jacobians = JointZeroCalibrator._center(active_jacobians)
        matrix = centered_jacobians.reshape(-1, active.size)
        return matrix, -centered_positions.reshape(-1)

    @staticmethod
    def _point_rms(positions: NDArray[np.float64]) -> float:
        """Return RMS Euclidean distance from the point centroid."""
        return _groups.point_rms(positions)

    @staticmethod
    def _point_errors(positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return stable Euclidean distances to the centroid, in sample order."""
        return _groups.point_errors(positions)

    def _validate_samples(self, samples: ArrayLike, unit: str) -> NDArray[np.float64]:
        return _groups.validate_samples(samples, unit, self.kinematics.dof)

    def _initial_offsets(self, value: ArrayLike | None) -> NDArray[np.float64]:
        offsets = (
            np.zeros(self.kinematics.dof)
            if value is None
            else real_array(value, "initial_offsets")
        )
        if offsets.shape != (self.kinematics.dof,) or not np.all(np.isfinite(offsets)):
            raise ValueError(
                f"initial_offsets must have shape ({self.kinematics.dof},) with finite values"
            )
        return offsets.copy()

    def _active_indices(self, fixed_joints: Sequence[int | str]) -> NDArray[np.int64]:
        names = self.kinematics.joint_names
        fixed: set[int] = set()
        if isinstance(fixed_joints, (str, bytes)):
            # Keep the public ValueError contract for invalid input values.
            raise ValueError(  # noqa: TRY004
                "fixed_joints must be a sequence of joint indices or names"
            )
        for joint in fixed_joints:
            if isinstance(joint, str):
                if joint not in names:
                    raise ValueError(f"unknown joint name: {joint}")
                fixed.add(names.index(joint))
            elif (
                isinstance(joint, (int, np.integer))
                and not isinstance(joint, (bool, np.bool_))
                and 0 <= int(joint) < self.kinematics.dof
            ):
                fixed.add(int(joint))
            else:
                raise ValueError(f"invalid fixed joint: {joint!r}")
        return np.asarray(
            [i for i in range(self.kinematics.dof) if i not in fixed], dtype=int
        )
