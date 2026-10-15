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


@dataclass(frozen=True)
class DistanceInformation:
    """Unchanged contact geometry and incident distance rows for candidates."""

    target_positions: NDArray[np.float64]
    target_jacobians: NDArray[np.float64]
    reference_positions: tuple[NDArray[np.float64], ...]
    position_residuals: tuple[NDArray[np.float64], ...]
    mean_jacobians: tuple[NDArray[np.float64], ...]
    plan: GroupPlan
    target: int
    total_samples: int
    incident: tuple[int, ...]

    def append(
        self, position: NDArray[np.float64], jacobian: NDArray[np.float64]
    ) -> tuple[tuple[int, NDArray[np.float64]], ...] | None:
        """Recompute only the changed group and distances touching it.

        Recenter the target observations exactly as the full matrix path does;
        incremental world-frame means would lose small separations or amplify
        roundoff near coincident centres.
        """
        if not self.incident:
            return ()
        positions = np.vstack((self.target_positions, position))
        jacobians = np.concatenate((self.target_jacobians, jacobian[None]))
        residuals = list(self.position_residuals)
        means = list(self.mean_jacobians)
        residuals[self.target] = center(positions)[0]
        means[self.target] = jacobians[0] - center(jacobians)[0]
        deltas, first_means, second_means, updates = [], [], [], []
        batched = len(self.incident) >= 3
        for index in self.incident:
            first, second = self.plan.endpoints[index]
            delta = _centre_displacement(
                self.reference_positions[first],
                self.reference_positions[second],
                residuals[first],
                residuals[second],
            )
            if batched:
                deltas.append(delta)
                first_means.append(means[first])
                second_means.append(means[second])
            else:
                row, _ = distance_row(
                    delta,
                    means[first],
                    means[second],
                    self.plan.distances[index],
                    self.total_samples,
                )
                if not np.all(np.isfinite(row)):
                    return None
                updates.append((index, row))
        if not batched:
            return tuple(updates)
        rows, _ = distance_rows(
            np.asarray(deltas),
            np.asarray(first_means),
            np.asarray(second_means),
            tuple(self.plan.distances[index] for index in self.incident),
            self.total_samples,
        )
        if not np.all(np.isfinite(rows)):
            return None
        return tuple(zip(self.incident, rows, strict=True))


@dataclass(frozen=True)
class CandidateInformation:
    """Cached square-root information for appending to one independent group."""

    matrix: NDArray[np.float64] | None
    reference_jacobian: NDArray[np.float64]
    reference_residual: NDArray[np.float64]
    update_scale: float
    matrix_rows: int
    sigma_m: float | None
    update_start: int
    distances: DistanceInformation | None = None

    def append(
        self, jacobian: NDArray[np.float64], position: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64] | None:
        """Return a compressed prospective matrix, or request full centring.

        The covariance of N centred rows gains N/(N+1) times the outer product
        of a new observation's displacement from the old mean. Apply this
        identity separately to each of the three Cartesian derivative rows.
        QR factors preserve singular values without forming normal equations.
        """
        if self.matrix is None:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            delta = (jacobian - self.reference_jacobian) + self.reference_residual
            update = weighted_values(delta, self.update_scale, self.sigma_m)
        if not np.all(np.isfinite(update)):
            # A first-row displacement can overflow even when a midpoint-based
            # full centring produces representable rows. Let the caller retry.
            return None
        if np.any((update != 0) & (np.abs(update) < np.finfo(float).tiny)):
            # Subnormal rounding can invalidate the covariance update identity.
            return None
        matrix = self.matrix.copy()
        matrix[self.update_start : self.update_start + 3] = update
        if self.distances is not None:
            assert position is not None
            updates = self.distances.append(position, jacobian)
            if updates is None:
                return None
            for index, row in updates:
                matrix[self.update_start + 3 + index] = row
        return matrix


def candidate_information(
    matrix: NDArray[np.float64],
    group_jacobians: Sequence[NDArray[np.float64]],
    plan: GroupPlan,
    active: NDArray[np.int64],
    target: int,
    *,
    group_positions: Sequence[NDArray[np.float64]] | None = None,
) -> CandidateInformation:
    """Compress group blocks and cache unchanged known-distance information."""
    total = sum(len(jacobians) for jacobians in group_jacobians)
    columns = active_columns(active, group_jacobians[0].shape[-1])
    count = len(group_jacobians[target])
    count_scale = np.sqrt((total + 1) / total)
    target_count_scale = np.sqrt(count / (count + 1))
    factors = []
    for index, section in enumerate(plan.sections):
        factor = np.linalg.qr(matrix[3 * section.start : 3 * section.stop], mode="r")
        with np.errstate(over="ignore", invalid="ignore"):
            factor *= count_scale
            if index == target:
                factor *= target_count_scale
        factors.append(factor)
    update_start = sum(len(factor) for factor in factors)
    factors.append(np.zeros((3, len(active))))
    distance_information = None
    if plan.distances:
        assert group_positions is not None
        referenced = {index for endpoints in plan.endpoints for index in endpoints}
        with np.errstate(over="ignore", invalid="ignore"):
            factors.append(matrix[-len(plan.distances) :] * count_scale)
        distance_information = DistanceInformation(
            group_positions[target],
            group_jacobians[target][:, :, columns].copy(order="K"),
            tuple(positions[0].copy() for positions in group_positions),
            tuple(
                center(positions)[0].copy() if index in referenced else np.zeros(3)
                for index, positions in enumerate(group_positions)
            ),
            tuple(
                (jacobians[:, :, columns][0] - center(jacobians[:, :, columns])[0]).copy()
                if index in referenced
                else np.zeros((3, len(active)))
                for index, jacobians in enumerate(group_jacobians)
            ),
            plan,
            target,
            total + 1,
            tuple(
                index
                for index, endpoints in enumerate(plan.endpoints)
                if target in endpoints
            ),
        )
    target_jacobians = group_jacobians[target][:, :, columns]
    update_scale = np.sqrt((total + 1) / (len(plan.groups) * (count + 1)))
    compressed = np.vstack(factors)
    # A QR norm may exceed float range even though measurement-normalized SVD
    # is finite. Preserve the full-matrix route in that exceptional case.
    return CandidateInformation(
        compressed
        if np.all(np.isfinite(compressed))
        and not np.any((matrix != 0) & (np.abs(matrix) < np.finfo(float).tiny))
        else None,
        target_jacobians[0].copy(),
        center(target_jacobians)[0].copy(),
        update_scale * target_count_scale,
        3 * (total + 1) + len(plan.distances),
        plan.groups[target].sigma_m,
        update_start,
        distance_information,
    )


def weighted_values(
    values: NDArray[np.float64], scale: float, sigma: float | None
) -> NDArray[np.float64]:
    """Apply a positive row scale and optional noise divisor without overflow.

    Keep the usual one-multiply path when its scalar coefficient is normal.
    For extreme coefficients, combine floating-point mantissas and exponents
    so an unrepresentable reciprocal cannot erase finite weighted rows.
    """
    if sigma is None:
        return values * scale
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        weight = scale / sigma
        if np.isfinite(weight) and weight >= np.finfo(float).tiny:
            return values * weight
        value_mantissa, value_exponent = np.frexp(values)
        scale_mantissa, scale_exponent = np.frexp(scale)
        sigma_mantissa, sigma_exponent = np.frexp(sigma)
        return np.ldexp(
            value_mantissa * scale_mantissa / sigma_mantissa,
            value_exponent + scale_exponent - sigma_exponent,
        )


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
    return centered_rms(center(positions))


def centered_rms(centered: NDArray[np.float64]) -> float:
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


def group_geometry(
    positions: NDArray[np.float64], plan: GroupPlan
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64] | None]:
    """Return centred positions and one centre displacement per distance."""
    groups = [positions[section] for section in plan.sections]
    centred = [center(group) for group in groups]
    return centred, centre_displacements(groups, centred, plan)


def centre_displacements(
    positions: Sequence[NDArray[np.float64]],
    centred: Sequence[NDArray[np.float64]],
    plan: GroupPlan,
) -> NDArray[np.float64] | None:
    """Subtract paired centroids before rounding them into world coordinates.

    Each distance has its own reference pair. A distant unrelated contact must
    neither erase a small spacing nor overflow an unused centre displacement.
    """
    if not plan.distances:
        return None
    return np.array(
        [
            _centre_displacement(
                positions[first][0],
                positions[second][0],
                centred[first][0],
                centred[second][0],
            )
            for first, second in plan.endpoints
        ]
    )


def _centre_displacement(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    first_residual: NDArray[np.float64],
    second_residual: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Subtract two inferred centroids using their representative-row residuals."""
    with np.errstate(over="ignore", invalid="ignore"):
        delta = (first - second) - first_residual + second_residual
    if not np.all(np.isfinite(delta)):
        terms = np.array([first, -second, -first_residual, second_residual])
        scale = np.max(np.abs(terms), axis=0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized = np.divide(
                terms, scale, out=np.zeros_like(terms), where=scale != 0
            )
            delta = normalized.sum(axis=0) * scale
    if not np.all(np.isfinite(delta)):
        raise ValueError("contact centres exceed the floating-point range")
    return delta


def distance_row(
    delta: NDArray[np.float64],
    first_mean: NDArray[np.float64],
    second_mean: NDArray[np.float64],
    distance: KnownDistance,
    total_samples: int,
) -> tuple[NDArray[np.float64], float]:
    """Return one weighted distance derivative and the predicted length."""
    length = distance_length(delta)
    if length == 0:
        raise CoincidentCentres(
            f"distance between {distance.first!r} and {distance.second!r} "
            "has coincident predicted centres; use distinct contacts or another offset seed"
        )
    direction = delta / length
    scale = np.sqrt(total_samples) * np.sqrt(distance.weight)
    roundoff_scale = 16 * np.finfo(float).eps
    with np.errstate(over="ignore", invalid="ignore"):
        derivative = direction @ (first_mean - second_mean)
        roundoff = np.abs(direction) @ (
            np.abs(first_mean) * roundoff_scale + np.abs(second_mean) * roundoff_scale
        )
        derivative[np.abs(derivative) <= roundoff] = 0.0
        row = weighted_values(derivative, scale, distance.sigma_m)
    failed = ~np.isfinite(row)
    if np.any(failed):
        # The mean difference or its projection can overflow before the noise
        # divisor brings the row back into range. Normalize affected columns.
        first, second = first_mean[:, failed], second_mean[:, failed]
        column_scale = np.maximum(
            np.max(np.abs(first), axis=0), np.max(np.abs(second), axis=0)
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            first, second = first / column_scale, second / column_scale
            projected = direction @ (first - second)
            floor = np.abs(direction) @ (
                np.abs(first) * roundoff_scale + np.abs(second) * roundoff_scale
            )
            projected[np.abs(projected) <= floor] = 0.0
            projected_mantissa, projected_exponent = np.frexp(projected)
            column_mantissa, column_exponent = np.frexp(column_scale)
            scale_mantissa, scale_exponent = np.frexp(scale)
            sigma_mantissa, sigma_exponent = np.frexp(
                distance.sigma_m if distance.sigma_m is not None else 1.0
            )
            row[failed] = np.ldexp(
                projected_mantissa * column_mantissa * scale_mantissa / sigma_mantissa,
                projected_exponent + column_exponent + scale_exponent - sigma_exponent,
            )
    return row, length


def distance_length(delta: NDArray[np.float64]) -> float:
    """Return a representable Euclidean distance from finite components."""
    with np.errstate(over="ignore", invalid="ignore"):
        length = float(np.hypot.reduce(delta))
    if not np.isfinite(length):
        raise ValueError("contact distance exceeds the floating-point range")
    return length


def distance_rows(
    deltas: NDArray[np.float64],
    first_means: NDArray[np.float64],
    second_means: NDArray[np.float64],
    distances: Sequence[KnownDistance],
    total_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project several distance derivatives together, with scalar rare fallbacks.

    Shapes are (K, 3) displacements and (K, 3, active_count) mean Jacobians.
    One or two distances keep the scalar path; larger sets share array
    operations. Preserve endpoint-specific errors and extreme row scaling.
    """
    if len(distances) < 3:
        if not distances:
            return np.empty((0, first_means.shape[-1])), np.empty(0)
        evaluated = [
            distance_row(delta, first, second, distance, total_samples)
            for delta, first, second, distance in zip(
                deltas, first_means, second_means, distances, strict=True
            )
        ]
        return np.array([row for row, _ in evaluated]), np.array(
            [length for _, length in evaluated]
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lengths = np.hypot.reduce(deltas, axis=1)
        invalid = np.flatnonzero((lengths == 0) | ~np.isfinite(lengths))
        if invalid.size:
            # Use the existing error and endpoint names for the first bad row.
            index = invalid[0]
            distance_row(
                deltas[index],
                first_means[index],
                second_means[index],
                distances[index],
                total_samples,
            )
        directions = deltas / lengths[:, None]
        derivatives = np.einsum("ij,ijk->ik", directions, first_means - second_means)
        roundoff_scale = 16 * np.finfo(float).eps
        roundoff = np.einsum(
            "ij,ijk->ik",
            np.abs(directions),
            np.abs(first_means) * roundoff_scale
            + np.abs(second_means) * roundoff_scale,
        )
        derivatives[np.abs(derivatives) <= roundoff] = 0
        scales = np.sqrt(total_samples) * np.sqrt(
            [distance.weight for distance in distances]
        )
        sigmas = np.array(
            [
                distance.sigma_m if distance.sigma_m is not None else 1.0
                for distance in distances
            ]
        )
        weights = scales / sigmas
        rows = derivatives * weights[:, None]
        for index in np.flatnonzero(
            ~np.isfinite(weights) | (weights < np.finfo(float).tiny)
        ):
            rows[index] = weighted_values(
                derivatives[index], scales[index], sigmas[index]
            )
        for index in np.flatnonzero(~np.all(np.isfinite(rows), axis=1)):
            rows[index], _ = distance_row(
                deltas[index],
                first_means[index],
                second_means[index],
                distances[index],
                total_samples,
            )
    return rows, lengths


def active_columns(active: NDArray[np.int64], width: int) -> slice | NDArray[np.int64]:
    """Use a view for consecutive active columns; preserve general indexing."""
    if (
        active.size
        and active.dtype.kind in "iu"
        and 0 <= active[0] <= active[-1] < width
        and active[-1] - active[0] + 1 == active.size
        and np.array_equal(active[1:], active[:-1] + 1)
    ):
        return slice(int(active[0]), int(active[-1]) + 1)
    return active


def _group_rows(
    centred_positions: Sequence[NDArray[np.float64]],
    group_jacobians: Sequence[NDArray[np.float64]],
    displacements: NDArray[np.float64] | None,
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
    columns = active_columns(active, group_jacobians[0].shape[-1])
    for group, positions, jacobians in zip(
        plan.groups, centred_positions, group_jacobians, strict=True
    ):
        jacobian = jacobians[:, :, columns]
        centred_jacobian = center(jacobian)
        if plan.distances:
            mean_jacobians.append(jacobian[0] - centred_jacobian[0])
        scale = np.sqrt(total_samples / (len(plan.groups) * len(positions)))
        rows.append(
            weighted_values(centred_jacobian, scale, group.sigma_m).reshape(
                -1, len(active)
            )
        )
        if with_residuals:
            residuals.append(weighted_values(positions, scale, group.sigma_m).ravel())
    if len(plan.distances) >= 3:
        assert displacements is not None
        distance_matrix, lengths = distance_rows(
            displacements,
            np.array([mean_jacobians[first] for first, _ in plan.endpoints]),
            np.array([mean_jacobians[second] for _, second in plan.endpoints]),
            plan.distances,
            total_samples,
        )
        rows.append(distance_matrix)
    elif plan.distances:
        assert displacements is not None
        lengths = np.empty(len(plan.distances))
        for index, (distance, (first, second)) in enumerate(
            zip(plan.distances, plan.endpoints, strict=True)
        ):
            row, lengths[index] = distance_row(
                displacements[index],
                mean_jacobians[first],
                mean_jacobians[second],
                distance,
                total_samples,
            )
            rows.append(row[None])
    if with_residuals:
        for index, distance in enumerate(plan.distances):
            # Multiplying before sqrt can overflow for representable rows.
            scale = np.sqrt(total_samples) * np.sqrt(distance.weight)
            residuals.append(
                weighted_values(
                    np.array([lengths[index] - distance.distance_m]),
                    scale,
                    distance.sigma_m,
                )
            )
    return rows, residuals


def group_state_system(
    positions: NDArray[np.float64],
    jacobians: NDArray[np.float64],
    plan: GroupPlan,
    active: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the weighted (matrix, residual) of one grouped evaluation."""
    centred, displacements = group_geometry(positions, plan)
    rows, residuals = _group_rows(
        centred,
        [jacobians[section] for section in plan.sections],
        displacements,
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
    # Position values only affect distance derivatives. Without distances,
    # the information matrix depends exclusively on centred Jacobians.
    referenced = {index for endpoints in plan.endpoints for index in endpoints}
    centred_positions = [
        center(positions) if index in referenced else positions
        for index, positions in enumerate(group_positions)
    ]
    displacements = centre_displacements(group_positions, centred_positions, plan)
    rows, _ = _group_rows(
        centred_positions,
        group_jacobians,
        displacements,
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
    centred, displacements = group_geometry(positions, plan)
    groups = tuple(
        GroupEvaluation(
            group.name,
            positions[section.start] - residual[0],
            _centered_errors(residual),
            centered_rms(residual),
            group.csv_rows,
            group.sigma_m,
        )
        for group, section, residual in zip(
            plan.groups, plan.sections, centred, strict=True
        )
    )
    distances = []
    for index, distance in enumerate(plan.distances):
        assert displacements is not None
        length = distance_length(displacements[index])
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
