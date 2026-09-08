"""Geometric (URDF origin) parameter calibration from contact groups.

Joint zero offsets and mounting-origin deviations are separate models: this
module fits deviations of selected URDF ``<origin>`` components only, using
the same grouped contact residuals, known distances and damped-SVD solver as
:mod:`point_zero_cal.calibrator`. Estimating both at once would leave an
unobservable equivalence between joint offsets and mounting rotations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _groups, _least_squares
from ._least_squares import backtracking_solve, damped_svd_step
from ._validation import (
    finite_real,
    finite_scalar,
    kinematics_metadata,
    positive_integer,
    real_array,
)
from .calibrator import Kinematics
from .contacts import ContactEvaluation, ContactGroup, KnownDistance

PARAMETER_COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class GeometricParameter:
    """One URDF origin component estimated as a deviation from nominal.

    ``joint_name`` names a revolute or single-level fixed joint in the URDF.
    ``component`` selects one origin axis. ``prior_scale`` is the one-sigma
    uncertainty of the deviation in the component's unit (metres for x/y/z,
    radians for roll/pitch/yaw). ``nominal`` defaults to zero because the
    estimate is a deviation from the URDF value. ``active`` controls whether
    the component participates in fitting. Optional ``lower``/``upper`` bound
    the accepted deviation, in the same unit.
    """

    joint_name: str
    component: str
    prior_scale: float
    nominal: float = 0.0
    active: bool = True
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.joint_name, str) or not self.joint_name.strip():
            raise ValueError("parameter joint_name must be a non-empty string")
        if (
            not isinstance(self.component, str)
            or self.component not in PARAMETER_COMPONENTS
        ):
            raise ValueError(
                f"parameter component must be one of {PARAMETER_COMPONENTS}"
            )
        if not isinstance(self.active, (bool, np.bool_)):
            raise ValueError("parameter active must be a boolean")  # noqa: TRY004
        prior_scale = finite_scalar(self.prior_scale, "prior_scale")
        nominal = finite_real(self.nominal, "nominal")
        lower = None if self.lower is None else finite_real(self.lower, "lower")
        upper = None if self.upper is None else finite_real(self.upper, "upper")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("parameter lower bound must not exceed upper bound")
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "prior_scale", prior_scale)
        object.__setattr__(self, "nominal", nominal)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def unit(self) -> str:
        """Metres for translation components and radians for rotations."""
        return "m" if self.component in ("x", "y", "z") else "rad"

    @property
    def label(self) -> str:
        """Joint and component joined with a dot, used in diagnostics."""
        return f"{self.joint_name}.{self.component}"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible parameter metadata with explicit units."""
        return {
            "joint_name": self.joint_name,
            "component": self.component,
            "unit": self.unit,
            "nominal": self.nominal,
            "prior_scale": self.prior_scale,
            "active": self.active,
            "lower": self.lower,
            "upper": self.upper,
        }


class GeometricKinematics(Kinematics, Protocol):
    """Kinematics plus TCP derivatives with respect to origin parameters.

    ``geometric_parameters`` lists the parameter model in the column order of
    the geometric Jacobian. Configuration and array conventions follow
    :class:`Kinematics`; Jacobian column ``k`` is the TCP position derivative
    with respect to parameter ``k``, in metres per parameter unit.
    """

    @property
    def geometric_parameters(self) -> tuple[GeometricParameter, ...]: ...

    def position_and_geometric_jacobian(
        self,
        configuration: NDArray[np.float64],
        parameters: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return TCP position (3,) and geometric Jacobian (3, n_parameters).

        ``parameters`` holds the current deviation of every geometric
        parameter, in parameter order and parameter units. Returned arrays may
        reuse backend buffers; callers must copy values before the next call.
        """
        ...


@dataclass
class _FitState:
    matrix: NDArray[np.float64]
    error: NDArray[np.float64]
    score: float
    positions: NDArray[np.float64]
    data_matrix: NDArray[np.float64]


@dataclass(frozen=True)
class GeometricParameterResult:
    """One estimated deviation with its definition and prior metadata."""

    joint_name: str
    component: str
    unit: str
    nominal: float
    estimate: float
    prior_scale: float
    active: bool
    lower: float | None
    upper: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_name": self.joint_name,
            "component": self.component,
            "unit": self.unit,
            "nominal": self.nominal,
            "estimate": self.estimate,
            "prior_scale": self.prior_scale,
            "active": self.active,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class GeometricCalibrationResult:
    """Immutable geometric calibration output.

    ``data_rank``, ``data_condition_number`` and ``singular_values`` describe
    the contact and distance rows only; prior rows are excluded so
    regularization cannot fake full observability. ``prior_penalty`` is the
    RMS of ``(estimate - nominal) / prior_scale`` over active parameters.
    ``nullspace_directions`` name below-cutoff parameter combinations in the
    data-only Jacobian, as ``"0.71*Joint2.pitch + 0.71*Joint5.pitch"``.
    """

    converged: bool
    message: str
    iterations: int
    residual_rms: float
    step_norm: float
    data_rank: int
    data_condition_number: float
    singular_values: NDArray[np.float64]
    prior_penalty: float
    parameters: tuple[GeometricParameterResult, ...]
    nullspace_directions: tuple[str, ...]
    initial_evaluation: ContactEvaluation | None = None
    evaluation: ContactEvaluation | None = None

    def __post_init__(self) -> None:
        values = np.array(self.singular_values, dtype=float, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "singular_values", values)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "converged": self.converged,
            "message": self.message,
            "iterations": self.iterations,
            "residual_rms": self.residual_rms,
            "step_norm": self.step_norm,
            "data_rank": self.data_rank,
            "data_condition_number": (
                self.data_condition_number
                if np.isfinite(self.data_condition_number)
                else None
            ),
            "singular_values": self.singular_values.tolist(),
            "prior_penalty": self.prior_penalty,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "nullspace_directions": list(self.nullspace_directions),
        }
        if self.initial_evaluation is not None:
            payload["initial_evaluation"] = self.initial_evaluation.to_dict()
        if self.evaluation is not None:
            payload["evaluation"] = self.evaluation.to_dict()
        return payload


class GeometricCalibrator:
    """Estimate URDF origin deviations from grouped contact observations.

    The objective mirrors the grouped zero-offset fit: separately centred
    contact groups (honouring each group's ``sigma_m``), optional known
    distances, plus one prior row per active parameter penalizing
    ``(deviation - nominal) / prior_scale``. Convergence additionally
    requires full rank of the data-only Jacobian, so priors can never hide
    unobservable parameter combinations. Solver options match
    :class:`~point_zero_cal.calibrator.JointZeroCalibrator`.
    """

    def __init__(
        self,
        kinematics: GeometricKinematics,
        *,
        max_iterations: int = 100,
        step_tolerance: float = 1e-8,
        residual_tolerance: float = 1e-6,
        damping: float = 1e-10,
        rank_tolerance: float = 1e-8,
    ) -> None:
        self.kinematics = kinematics
        kinematics_metadata(kinematics.dof, kinematics.joint_names)
        parameters = tuple(kinematics.geometric_parameters)
        if not parameters:
            raise ValueError("kinematics must define at least one geometric parameter")
        if any(
            not isinstance(parameter, GeometricParameter) for parameter in parameters
        ):
            raise ValueError(
                "geometric_parameters must contain GeometricParameter values"
            )
        labels = [parameter.label for parameter in parameters]
        if len(set(labels)) != len(labels):
            raise ValueError("geometric parameter labels must be unique")
        self._parameters = parameters
        self.max_iterations = positive_integer(max_iterations, "max_iterations")
        self.step_tolerance = finite_scalar(step_tolerance, "step_tolerance")
        self.residual_tolerance = finite_scalar(
            residual_tolerance, "residual_tolerance"
        )
        self.damping = finite_scalar(damping, "damping", allow_zero=True)
        self.rank_tolerance = finite_scalar(rank_tolerance, "rank_tolerance")
        if self.rank_tolerance >= 1:
            raise ValueError("rank_tolerance must be less than 1")

    @property
    def parameter_labels(self) -> tuple[str, ...]:
        """Parameter labels in geometric Jacobian column order."""
        return tuple(parameter.label for parameter in self._parameters)

    def calibrate_groups(
        self,
        groups: Sequence[ContactGroup],
        *,
        distances: Sequence[KnownDistance] = (),
        sample_unit: str = "rad",
        initial_parameters: ArrayLike | None = None,
        distance_tolerance: float | None = None,
    ) -> GeometricCalibrationResult:
        """Fit shared origin deviations to separately centred contact groups.

        The squared objective is the mean of group point-RMS squares (divided
        by each group's ``sigma_m`` when set), plus weighted distance errors
        (divided by ``sigma_m``), plus squared normalized prior residuals over
        active parameters. Accepted candidates are projected onto optional
        parameter bounds.

        Convergence requires a non-stalled, small remaining correction, every
        group's RMS within ``residual_tolerance``, every unweighted distance
        error within ``distance_tolerance`` (defaults to residual tolerance),
        and full rank of the data-only Jacobian.
        """
        q, plan = self._prepare_groups(groups, distances, sample_unit)
        parameters = self._initial_parameters(initial_parameters)
        active = self._active_indices()
        if not active.size:
            raise ValueError("at least one parameter must remain calibratable")
        distance_tolerance = (
            self.residual_tolerance
            if distance_tolerance is None
            else finite_scalar(distance_tolerance, "distance_tolerance")
        )
        x0 = parameters[active].copy()

        def evaluate(x: NDArray[np.float64]) -> _FitState:
            full = parameters.copy()
            full[active] = x
            return self._state(full, q, plan, active)

        def apply_step(
            x: NDArray[np.float64], step: NDArray[np.float64]
        ) -> NDArray[np.float64] | None:
            return self._project_bounds(x + step, active)

        fit = backtracking_solve(
            x0,
            evaluate,
            apply_step,
            max_iterations=self.max_iterations,
            step_tolerance=self.step_tolerance,
            damping=self.damping,
            rank_tolerance=self.rank_tolerance,
        )
        full = parameters.copy()
        full[active] = fit.parameters
        initial_positions = fit.initial.positions
        positions = fit.current.positions
        data_matrix = fit.current.data_matrix
        # Wide systems need the complete right basis to include the structural
        # nullspace. Keep the reduced SVD for tall sample matrices to avoid a
        # quadratic allocation in the number of observations.
        _, values, right = np.linalg.svd(
            data_matrix, full_matrices=data_matrix.shape[0] < data_matrix.shape[1]
        )
        cutoff = values[0] * self.rank_tolerance if values.size else 0.0
        data_rank = int(np.count_nonzero(values > cutoff))
        condition = (
            float(values[0] / values[-1]) if data_rank == active.size else float("inf")
        )
        nullspace = self._nullspace_directions(right[data_rank:], active)
        initial = _groups.group_report(
            initial_positions, plan, np.zeros(self.kinematics.dof)
        )
        final = _groups.group_report(positions, plan, np.zeros(self.kinematics.dof))
        prior_residuals = np.array(
            [
                (full[k] - self._parameters[k].nominal)
                / self._parameters[k].prior_scale
                for k in active
            ]
        )
        prior_penalty = float(
            np.hypot.reduce(prior_residuals / np.sqrt(len(active)))
            if len(prior_residuals)
            else 0.0
        )
        if data_rank < active.size:
            message = (
                f"insufficient data observability: rank {data_rank} < "
                f"{active.size} active parameters"
            )
        elif any(group.point_rms_m > self.residual_tolerance for group in final.groups):
            message = "contact group residual did not reach tolerance"
        elif any(
            abs(distance.error_m) > distance_tolerance for distance in final.distances
        ):
            message = "distance residual did not reach tolerance"
        elif fit.stalled:
            message = self._stalled_message(fit, active)
        elif fit.step_norm > self.step_tolerance:
            message = "maximum iterations reached before the parameter step converged"
        else:
            message = "geometric calibration converged"
        return GeometricCalibrationResult(
            converged=message == "geometric calibration converged",
            message=message,
            iterations=fit.iterations,
            residual_rms=final.residual_rms_m,
            step_norm=fit.step_norm,
            data_rank=data_rank,
            data_condition_number=condition,
            singular_values=values,
            prior_penalty=prior_penalty,
            parameters=tuple(
                GeometricParameterResult(
                    parameter.joint_name,
                    parameter.component,
                    parameter.unit,
                    parameter.nominal,
                    full[k],
                    parameter.prior_scale,
                    parameter.active,
                    parameter.lower,
                    parameter.upper,
                )
                for k, parameter in enumerate(self._parameters)
            ),
            nullspace_directions=nullspace,
            initial_evaluation=initial,
            evaluation=final,
        )

    def evaluate_groups(
        self,
        groups: Sequence[ContactGroup],
        parameters: ArrayLike,
        *,
        distances: Sequence[KnownDistance] = (),
        sample_unit: str = "rad",
    ) -> ContactEvaluation:
        """Evaluate fixed origin deviations on held-out groups, without fitting.

        Group centroids are inferred from these samples; the report measures
        contact consistency and relative distances, not absolute world
        accuracy. ``ContactEvaluation.offsets_rad`` is a zero vector because
        group consistency does not depend on joint zero offsets.
        """
        q, plan = self._prepare_groups(groups, distances, sample_unit)
        values = real_array(parameters, "parameters")
        if values.shape != (len(self._parameters),) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"parameters must have shape ({len(self._parameters)},) with finite values"
            )
        positions, _ = self._evaluate(q, values.copy())
        return _groups.group_report(positions, plan, np.zeros(self.kinematics.dof))

    def _prepare_groups(
        self,
        groups: Sequence[ContactGroup],
        distances: Sequence[KnownDistance],
        unit: str,
    ) -> tuple[NDArray[np.float64], _groups.GroupPlan]:
        return _groups.prepare_groups(groups, distances, unit, self.kinematics.dof)

    def _initial_parameters(self, value: ArrayLike | None) -> NDArray[np.float64]:
        parameters = np.array(
            [parameter.nominal for parameter in self._parameters], dtype=float
        )
        if value is not None:
            seed = real_array(value, "initial_parameters")
            if seed.shape != (len(self._parameters),) or not np.all(np.isfinite(seed)):
                raise ValueError(
                    f"initial_parameters must have shape ({len(self._parameters)},) "
                    "with finite values"
                )
            parameters = seed.copy()
        for parameter, estimate in zip(self._parameters, parameters, strict=True):
            if (
                parameter.lower is not None and estimate < parameter.lower
            ) or (parameter.upper is not None and estimate > parameter.upper):
                raise ValueError(
                    f"initial_parameters for {parameter.label} must lie within bounds"
                )
        return parameters

    def _active_indices(self) -> NDArray[np.int64]:
        return np.asarray(
            [k for k, parameter in enumerate(self._parameters) if parameter.active],
            dtype=int,
        )

    def _project_bounds(
        self, candidate: NDArray[np.float64], active: NDArray[np.int64]
    ) -> NDArray[np.float64] | None:
        projected = candidate.copy()
        for row, k in enumerate(active):
            parameter = self._parameters[k]
            if parameter.lower is not None and projected[row] < parameter.lower:
                projected[row] = parameter.lower
            if parameter.upper is not None and projected[row] > parameter.upper:
                projected[row] = parameter.upper
        if not np.all(np.isfinite(projected)):
            return None
        return projected

    def _state(
        self,
        parameters: NDArray[np.float64],
        q: NDArray[np.float64],
        plan: _groups.GroupPlan,
        active: NDArray[np.int64],
    ) -> _FitState:
        positions, jacobians = self._evaluate(q, parameters)
        data_matrix, data_residual = _groups.group_state_system(
            positions, jacobians, plan, active
        )
        matrix, residual = self._with_prior_rows(
            parameters, active, len(q), data_matrix, data_residual
        )
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(residual)):
            raise ValueError(
                "weighted contact residuals exceed the floating-point range"
            )
        score = float(np.hypot.reduce(residual / np.sqrt(len(q))))
        if not np.isfinite(score):
            raise ValueError("contact objective exceeds the floating-point range")
        return _FitState(matrix, -residual, score, positions, data_matrix)

    def _with_prior_rows(
        self,
        parameters: NDArray[np.float64],
        active: NDArray[np.int64],
        total_samples: int,
        data_matrix: NDArray[np.float64],
        data_residual: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Append one normalized prior row per active parameter.

        Rows use the same ``sqrt(total_samples)`` scale as distance rows, so
        each prior contributes ``((x - nominal) / prior_scale)**2`` to the
        squared objective regardless of sample count. Priors never enter
        observability diagnostics, which use the data rows only.
        """
        rows = [data_matrix]
        residuals = [data_residual]
        prior_matrix = np.zeros((len(active), len(active)))
        prior_residual = np.empty(len(active))
        for row, k in enumerate(active):
            scale = np.sqrt(total_samples) / self._parameters[k].prior_scale
            prior_matrix[row, row] = scale
            prior_residual[row] = (parameters[k] - self._parameters[k].nominal) * scale
        rows.append(prior_matrix)
        residuals.append(prior_residual)
        return np.vstack(rows), np.concatenate(residuals)

    def _stalled_message(
        self, fit: _least_squares.FitResult, active: NDArray[np.int64]
    ) -> str:
        """Attribute a stall to bounds when projection blocks all movement."""
        step, _, _ = damped_svd_step(
            fit.current.matrix,
            fit.current.error,
            damping=self.damping,
            rank_tolerance=self.rank_tolerance,
        )
        projected = self._project_bounds(fit.parameters + step, active)
        if projected is not None and np.all(projected == fit.parameters):
            return "parameter bound blocked the remaining correction"
        return "no residual-decreasing parameter step found"

    def _nullspace_directions(
        self, vectors: NDArray[np.float64], active: NDArray[np.int64]
    ) -> tuple[str, ...]:
        """Name below-cutoff parameter combinations from right singular vectors."""
        directions = []
        labels = [self._parameters[k].label for k in active]
        for vector in vectors:
            scale = float(np.max(np.abs(vector)))
            if not np.isfinite(scale) or scale <= 0:
                continue
            terms = [
                f"{coefficient:.4g}*{label}"
                for coefficient, label in zip(vector, labels, strict=True)
                if abs(coefficient) >= 0.3 * scale
            ]
            if terms:
                directions.append(" + ".join(terms))
        return tuple(directions)

    def _evaluate(
        self, q: NDArray[np.float64], parameters: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate and validate the geometric backend at every configuration."""
        positions = np.empty((len(q), 3))
        jacobians = np.empty((len(q), 3, len(self._parameters)))
        for index, configuration in enumerate(q):
            position, jacobian = self.kinematics.position_and_geometric_jacobian(
                configuration, parameters
            )
            position = real_array(position, "kinematics position")
            jacobian = real_array(jacobian, "kinematics geometric jacobian")
            if position.shape != (3,) or jacobian.shape != (
                3,
                len(self._parameters),
            ):
                raise ValueError(
                    "kinematics must return position shape (3,) and geometric "
                    f"jacobian shape (3, {len(self._parameters)})"
                )
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(jacobian)):
                raise ValueError("kinematics returned non-finite values")
            positions[index] = position
            jacobians[index] = jacobian
        return positions, jacobians
