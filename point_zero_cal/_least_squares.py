"""Shared damped-SVD least-squares core with halving backtracking.

The zero-offset and geometric calibrators both minimize weighted residual
norms through the same damped SVD steps and step-halving line search. The
solver is generic over an active-space parameter vector; callers map a
candidate back onto their own model through ``evaluate`` and ``apply_step``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from math import ceil, log10
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

_TINY_NORMAL = float(np.finfo(float).tiny)


class LeastSquaresState(Protocol):
    """Linearized residual and objective required by the shared solver."""

    matrix: NDArray[np.float64]
    error: NDArray[np.float64]
    score: float


StateT = TypeVar("StateT", bound=LeastSquaresState)


class CoincidentCentres(ValueError):
    """A positive-distance residual has no derivative at coincident centres."""


@dataclass
class FitResult(Generic[StateT]):
    """Accepted parameter vector with the initial and final solver states."""

    parameters: NDArray[np.float64]
    initial: StateT
    current: StateT
    iterations: int
    step_norm: float
    singular_values: NDArray[np.float64]
    stalled: bool


def numerical_rank(values: NDArray[np.float64], tolerance: float) -> int:
    """Count observable directions in a descending, finite SVD spectrum.

    Normalize before comparing to avoid rounding a subnormal cutoff up to
    the largest singular value and incorrectly losing every direction.
    """
    if not values.size or values[0] == 0:
        return 0
    return int(np.count_nonzero(values / values[0] > tolerance))


def _decimal_right_product(
    right: NDArray[np.float64],
    mantissas: NDArray[np.float64],
    exponents: NDArray[np.int32],
) -> list[Decimal]:
    """Multiply scaled coefficients with cancellation preserved in Decimal.

    A singular coefficient can exceed float range while its contribution to
    each parameter remains representable. Decimal arithmetic is confined to
    this exceptional path and also retains tiny modes after large cancellation.
    Precision follows the exponent span, including the right-basis weights.
    """
    selected = np.flatnonzero(mantissas)
    if not selected.size:
        return [Decimal(0)] * right.shape[1]
    weights = right[selected]
    weight_exponents = np.frexp(weights[weights != 0])[1]
    if not weight_exponents.size:
        return [Decimal(0)] * right.shape[1]
    coefficient_exponents = exponents[selected]
    span = (
        int(coefficient_exponents.max())
        - int(coefficient_exponents.min())
        + int(weight_exponents.max())
        - int(weight_exponents.min())
    )
    result = []
    # Caller Decimal settings must not constrain this float solver's fallback.
    context = Context(
        prec=80 + ceil(span * log10(2)),
        rounding=ROUND_HALF_EVEN,
        Emin=-999999,
        Emax=999999,
        clamp=0,
        traps=[],
        flags=[],
    )
    with localcontext(context):
        powers = None
        if selected.size >= 32:
            unique_exponents = np.unique(coefficient_exponents)
            if 2 * len(unique_exponents) < selected.size:
                powers = {
                    int(exponent): Decimal(2) ** int(exponent)
                    for exponent in unique_exponents
                }
        coefficients = [
            Decimal.from_float(float(mantissas[index]))
            * (
                Decimal(2) ** int(exponents[index])
                if powers is None
                else powers[int(exponents[index])]
            )
            for index in selected
        ]
        for column in weights.T:
            value = sum(
                (
                    Decimal.from_float(float(weight)) * coefficient
                    for weight, coefficient in zip(column, coefficients, strict=True)
                    if weight != 0
                ),
                Decimal(0),
            )
            result.append(value)
    return result


def _scaled_right_product(
    right: NDArray[np.float64],
    mantissas: NDArray[np.float64],
    exponents: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Map scaled singular coefficients into parameter coordinates."""
    return np.array(
        [float(value) for value in _decimal_right_product(right, mantissas, exponents)]
    )


def _scaled_left_projection(
    left: NDArray[np.float64], error: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Recover extreme projections without losing small residual components."""
    mantissas, exponents = np.frexp(error)
    values = _decimal_right_product(left, mantissas, exponents)
    projected_mantissas = np.zeros(left.shape[1])
    projected_exponents = np.zeros(left.shape[1], dtype=np.int32)
    for index, value in enumerate(values):
        if not value:
            continue
        numerator, denominator = value.as_integer_ratio()
        exponent = abs(numerator).bit_length() - denominator.bit_length() + 1
        projected_mantissas[index] = (
            numerator / (denominator << exponent)
            if exponent >= 0
            else (numerator << -exponent) / denominator
        )
        projected_exponents[index] = exponent
    return projected_mantissas, projected_exponents


def damped_svd_step(
    matrix: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    damping: float,
    rank_tolerance: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Return a damped step, undamped remaining-step norm, and spectrum."""
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    if not np.isfinite(values[0]):
        raise ValueError("linear solve spectrum exceeds the floating-point range")
    rank = numerical_rank(values, rank_tolerance)
    # SVD values are descending, so observable directions form a prefix.
    # Slices retain views instead of copying the tall left basis with a mask.
    left = left[:, :rank]
    right = right[:rank]
    scales = values[:rank]
    # Divide the projected residual directly: forming 1/s first can
    # overflow even when the resulting correction is representable.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        projected = left.T @ error
        correction = projected / scales
        remaining = right.T @ correction
        tiny_projections = None
        if projected.size and np.abs(projected).min() < _TINY_NORMAL and np.any(error):
            tiny_projections = np.abs(projected) < _TINY_NORMAL
        if not np.all(np.isfinite(remaining)) or (
            tiny_projections is not None and np.any(tiny_projections)
        ):
            # Projection can overflow, or underflow before division by a tiny
            # singular value restores a representable correction.
            failed = ~np.isfinite(correction)
            if tiny_projections is not None:
                failed |= tiny_projections
            projected_mantissa, projected_exponent = np.frexp(projected[failed])
            recover = ~np.isfinite(projected[failed])
            if tiny_projections is not None:
                recover |= tiny_projections[failed]
            if np.any(recover):
                projected_mantissa[recover], projected_exponent[recover] = (
                    _scaled_left_projection(left[:, failed][:, recover], error)
                )
            singular_mantissa, singular_exponent = np.frexp(scales[failed])
            # Separate mantissas/exponents so neither division nor restoring
            # the residual scale loses a representable tiny correction.
            # Keep finite modes: globally scaling the error can underflow
            # smaller components that were already projected successfully.
            mantissas, exponents = np.frexp(correction)
            mantissas[failed] = projected_mantissa / singular_mantissa
            exponents[failed] = projected_exponent - singular_exponent
            remaining = _scaled_right_product(right, mantissas, exponents)
            if not np.all(np.isfinite(remaining)):
                raise ValueError("linear solve produced a non-finite offset step")
            if damping == 0:
                step = remaining.copy()
            else:
                attenuation = scales / np.hypot(scales, np.sqrt(damping))
                attenuation_mantissa, attenuation_exponent = np.frexp(attenuation)
                step = (
                    _scaled_right_product(
                        right,
                        (mantissas * attenuation_mantissa) * attenuation_mantissa,
                        exponents + 2 * attenuation_exponent,
                    )
                    if np.any(attenuation)
                    else np.zeros(right.shape[1])
                )
        # hypot avoids squaring dimensional singular values, which can
        # overflow or underflow for large or small kinematic length scales.
        else:
            if damping == 0:
                step = remaining.copy()
            else:
                attenuation = scales / np.hypot(scales, np.sqrt(damping))
                # Apply factors separately: their square may underflow while the
                # product with a large undamped correction is representable.
                step = right.T @ ((correction * attenuation) * attenuation)
    if not np.all(np.isfinite(step)):
        raise ValueError("linear solve produced a non-finite offset step")
    return step, float(np.linalg.norm(remaining, ord=np.inf)), values


def backtracking_solve(
    x0: NDArray[np.float64],
    evaluate: Callable[[NDArray[np.float64]], StateT],
    apply_step: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64] | None
    ],
    *,
    max_iterations: int,
    step_tolerance: float,
    damping: float,
    rank_tolerance: float,
) -> FitResult[StateT]:
    """Damped SVD iterations with halving backtracking over the active space.

    ``evaluate`` returns a state exposing ``matrix``, ``error`` and ``score``
    and may raise :class:`CoincidentCentres` when a distance residual has no
    derivative, in which case a shorter step is tried. ``apply_step`` maps a
    candidate vector and a trial step to the next candidate, or returns
    ``None`` to skip a trial that left its valid domain.
    """
    initial = current = evaluate(np.asarray(x0, dtype=float))
    step, step_norm, singular_values = damped_svd_step(
        current.matrix,
        current.error,
        damping=damping,
        rank_tolerance=rank_tolerance,
    )
    parameters = np.array(x0, dtype=float, copy=True)
    stalled = False
    for iteration in range(1, max_iterations + 1):  # noqa: B007 - reported in FitResult
        small_step = step_norm <= step_tolerance
        if not np.any(step):
            stalled = not small_step
            break
        previous_candidate = None
        for backtrack in range(1 if small_step else 21):
            candidate = apply_step(parameters, 0.5**backtrack * step)
            if candidate is None or not np.all(np.isfinite(candidate)):
                continue
            if np.array_equal(candidate, parameters) or (
                previous_candidate is not None
                and np.array_equal(candidate, previous_candidate)
            ):
                continue
            previous_candidate = candidate.copy()
            try:
                trial = evaluate(candidate)
            except CoincidentCentres:
                # A shorter step can avoid this undefined distance derivative.
                continue
            if trial.score < current.score:
                parameters, current = candidate, trial
                step, step_norm, singular_values = damped_svd_step(
                    current.matrix,
                    current.error,
                    damping=damping,
                    rank_tolerance=rank_tolerance,
                )
                break
        else:
            stalled = not small_step
            break
        if small_step:
            break
    return FitResult(
        parameters, initial, current, iteration, step_norm, singular_values, stalled
    )
