"""Shared contact-group planning, centring and weighted residual rows."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._least_squares import CoincidentCentres
from ._validation import real_array
from .contacts import (
    ContactEvaluation,
    ContactGroup,
    DistanceEvaluation,
    GroupEvaluation,
    KnownDistance,
)


@dataclass(frozen=True)
class GroupPlan:
    """Contact groups, stacked-sample sections and resolved distance endpoints."""

    groups: tuple[ContactGroup, ...]
    sections: tuple[slice, ...]
    distances: tuple[KnownDistance, ...]
    endpoints: tuple[tuple[int, int], ...]


def validate_samples(samples: ArrayLike, unit: str, dof: int) -> NDArray[np.float64]:
    """Validate one group's samples and convert the unit to radians."""
    q = real_array(samples, "samples")
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != dof:
        raise ValueError(f"samples must have shape (N, {dof}) with N >= 2")
    if not np.all(np.isfinite(q)):
        raise ValueError("samples must contain only finite values")
    if unit == "deg":
        return np.deg2rad(q)
    if unit != "rad":
        raise ValueError("sample_unit must be 'rad' or 'deg'")
    return q.copy()


def prepare_groups(
    groups: Sequence[ContactGroup],
    distances: Sequence[KnownDistance],
    unit: str,
    dof: int,
) -> tuple[NDArray[np.float64], GroupPlan]:
    """Validate observations and stack every group's samples in radians."""
    groups, distances = tuple(groups), tuple(distances)
    if not groups or any(not isinstance(group, ContactGroup) for group in groups):
        raise ValueError("groups must contain ContactGroup observations")
    names = [group.name for group in groups]
    if len(set(names)) != len(names):
        raise ValueError("contact group names must be unique")
    indices = {name: index for index, name in enumerate(names)}
    configurations, sections = [], []
    offset = 0
    for group in groups:
        q = validate_samples(group.samples, unit, dof)
        configurations.append(q)
        sections.append(slice(offset, offset + len(q)))
        offset += len(q)
    endpoints = []
    seen = set()
    for distance in distances:
        if not isinstance(distance, KnownDistance):
            raise ValueError("distances must contain KnownDistance measurements")  # noqa: TRY004
        if distance.first not in indices or distance.second not in indices:
            raise ValueError("distance references an unknown contact group")
        key = frozenset((distance.first, distance.second))
        if key in seen:
            raise ValueError("duplicate distance for a contact group pair")
        seen.add(key)
        endpoints.append((indices[distance.first], indices[distance.second]))
    return np.vstack(configurations), GroupPlan(
        groups, tuple(sections), distances, tuple(endpoints)
    )


def center(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center rows without subtracting a rounded large common baseline."""
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = values - values[0]
        # Divide before summation to avoid overflowing a representable mean.
        centered = shifted - (shifted / len(values)).sum(axis=0)
    if not np.all(np.isfinite(centered)):
        # Opposite near-limit values can overflow relative to the first row
        # although the residuals fit. Keep this fallback off the normal path.
        if np.all(np.isfinite(values)):
            with np.errstate(over="ignore", invalid="ignore"):
                midpoint = values.min(axis=0) / 2 + values.max(axis=0) / 2
                shifted = values - midpoint
                centered = shifted - (shifted / len(values)).sum(axis=0)
        if not np.all(np.isfinite(centered)):
            raise ValueError("centered kinematics exceed the floating-point range")
    return centered


def point_rms(positions: NDArray[np.float64]) -> float:
    """Return RMS Euclidean distance from the point centroid."""
    return _centered_rms(center(positions))


def _centered_rms(centered: NDArray[np.float64]) -> float:
    """Measure already-centred residuals without repeating the reduction."""
    scale = float(np.max(np.abs(centered)))
    if scale == 0:
        return 0.0
    normalized = centered / scale
    rms = scale * float(np.sqrt(np.mean(np.sum(normalized**2, axis=1))))
    if not np.isfinite(rms):
        raise ValueError("point RMS exceeds the floating-point range")
    return rms


def point_errors(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return stable Euclidean distances to the centroid, in sample order."""
    return _centered_errors(center(positions))


def _centered_errors(centered: NDArray[np.float64]) -> NDArray[np.float64]:
    """Measure already-centred residuals with stable Euclidean norms."""
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.hypot.reduce(centered, axis=1)
    if not np.all(np.isfinite(errors)):
        raise ValueError("point errors exceed the floating-point range")
    return errors


def group_centres(
    positions: NDArray[np.float64], plan: GroupPlan
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Return per-group centred positions and their common-origin centres."""
    centred = [center(positions[section]) for section in plan.sections]
    # Use a common origin so large world coordinates do not lose the
    # displacement between separately rounded world-frame centroids.
    with np.errstate(over="ignore", invalid="ignore"):
        relative = np.array(
            [
                (positions[section.start] - positions[0]) - residual[0]
                for section, residual in zip(plan.sections, centred, strict=True)
            ]
        )
    if not np.all(np.isfinite(relative)):
        raise ValueError("contact centres exceed the floating-point range")
    return centred, relative


def _group_rows(
    centred_positions: list[NDArray[np.float64]],
    group_jacobians: Sequence[NDArray[np.float64]],
    centres: NDArray[np.float64],
    plan: GroupPlan,
    active: NDArray[np.int64],
    total_samples: int,
    *,
    with_residuals: bool,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Build weighted group and distance rows, optionally with residual values.

    Group blocks are centred per contact and scaled so each group contributes
    equally regardless of sample count; a group's ``sigma_m`` downweights its
    consistency residual. Distance rows use the same derivative with roundoff
    suppression, scaled by ``sqrt(weight)`` and ``sigma_m``.
    """
    rows, residuals, mean_jacobians = [], [], []
    for group, positions, jacobians in zip(
        plan.groups, centred_positions, group_jacobians, strict=True
    ):
        jacobian = jacobians[:, :, active]
        centred_jacobian = center(jacobian)
        mean_jacobians.append(jacobian[0] - centred_jacobian[0])
        scale = np.sqrt(total_samples / (len(plan.groups) * len(positions)))
        if group.sigma_m is not None:
            # A noisier group's consistency residual counts less.
            scale /= group.sigma_m
        rows.append((centred_jacobian * scale).reshape(-1, len(active)))
        if with_residuals:
            residuals.append((positions * scale).ravel())
    for distance, (first, second) in zip(plan.distances, plan.endpoints, strict=True):
        delta = centres[first] - centres[second]
        length = float(np.hypot.reduce(delta))
        if length == 0:
            raise CoincidentCentres(
                f"distance between {distance.first!r} and {distance.second!r} "
                "has coincident predicted centres; use distinct contacts or another offset seed"
            )
        direction = delta / length
        derivative = direction @ (mean_jacobians[first] - mean_jacobians[second])
        # Suppress dot-product cancellation at roundoff level, especially
        # a common rigid rotation which cannot change centre distance.
        roundoff = (
            16
            * np.finfo(float).eps
            * (
                np.abs(direction)
                @ (np.abs(mean_jacobians[first]) + np.abs(mean_jacobians[second]))
            )
        )
        derivative[np.abs(derivative) <= roundoff] = 0.0
        # Multiplying before the square root can overflow even when every
        # weighted row remains representable.
        scale = np.sqrt(total_samples) * np.sqrt(distance.weight)
        if distance.sigma_m is not None:
            scale /= distance.sigma_m
        rows.append((derivative * scale)[None, :])
        if with_residuals:
            residuals.append(np.array([(length - distance.distance_m) * scale]))
    return rows, residuals


def group_state_system(
    positions: NDArray[np.float64],
    jacobians: NDArray[np.float64],
    plan: GroupPlan,
    active: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the weighted (matrix, residual) of one grouped evaluation."""
    centred, centres = group_centres(positions, plan)
    rows, residuals = _group_rows(
        centred,
        [jacobians[section] for section in plan.sections],
        centres,
        plan,
        active,
        len(positions),
        with_residuals=True,
    )
    return np.vstack(rows), np.concatenate(residuals)


def contact_information_matrix(
    group_positions: Sequence[NDArray[np.float64]],
    group_jacobians: Sequence[NDArray[np.float64]],
    plan: GroupPlan,
    active: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Build the grouped residual Jacobian without evaluating residual values."""
    if len(group_positions) != len(plan.groups) or len(group_jacobians) != len(
        plan.groups
    ):
        raise ValueError("contact groups do not match the calibration plan")
    total_samples = sum(len(positions) for positions in group_positions)
    if total_samples < 2:
        raise ValueError("contact groups must contain at least two samples")
    centred_positions = [center(positions) for positions in group_positions]
    anchor = group_positions[0][0]
    with np.errstate(over="ignore", invalid="ignore"):
        centres = np.array(
            [
                (positions[0] - anchor) - centred[0]
                for positions, centred in zip(
                    group_positions, centred_positions, strict=True
                )
            ]
        )
    if not np.all(np.isfinite(centres)):
        raise ValueError("contact centres exceed the floating-point range")
    rows, _ = _group_rows(
        centred_positions,
        group_jacobians,
        centres,
        plan,
        active,
        total_samples,
        with_residuals=False,
    )
    matrix = np.vstack(rows)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weighted contact Jacobian exceeds the floating-point range")
    return matrix


def group_report(
    positions: NDArray[np.float64], plan: GroupPlan, offsets: NDArray[np.float64]
) -> ContactEvaluation:
    """Build per-group and per-distance diagnostics from evaluated positions."""
    centred, centres = group_centres(positions, plan)
    groups = tuple(
        GroupEvaluation(
            group.name,
            positions[section.start] - residual[0],
            _centered_errors(residual),
            _centered_rms(residual),
            group.csv_rows,
            group.sigma_m,
        )
        for group, section, residual in zip(
            plan.groups, plan.sections, centred, strict=True
        )
    )
    distances = []
    for distance, (a, b) in zip(plan.distances, plan.endpoints, strict=True):
        length = float(np.hypot.reduce(centres[a] - centres[b]))
        distances.append(
            DistanceEvaluation(
                distance.first,
                distance.second,
                distance.distance_m,
                length,
                length - distance.distance_m,
                distance.weight,
                distance.sigma_m,
            )
        )
    point_rms_value = float(
        np.hypot.reduce(
            np.array([g.point_rms_m for g in groups]) / np.sqrt(len(groups))
        )
    )
    residual_rms = float(
        np.hypot.reduce(
            np.array(
                [
                    point_rms_value,
                    *(np.sqrt(d.weight) * d.error_m for d in distances),
                ]
            )
        )
    )
    if not np.isfinite(residual_rms):
        raise ValueError("contact objective exceeds the floating-point range")
    return ContactEvaluation(
        offsets, groups, tuple(distances), point_rms_value, residual_rms
    )
