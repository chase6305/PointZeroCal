"""Shared damped-SVD least-squares core with halving backtracking.

The zero-offset and geometric calibrators both minimize weighted residual
norms through the same damped SVD steps and step-halving line search. The
solver is generic over an active-space parameter vector; callers map a
candidate back onto their own model through ``evaluate`` and ``apply_step``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray


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


def damped_svd_step(
    matrix: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    damping: float,
    rank_tolerance: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Return a damped step, undamped remaining-step norm, and spectrum."""
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    observable = values > values[0] * rank_tolerance
    scales = values[observable]
    # Divide the projected residual directly: forming 1/s first can
    # overflow even when the resulting correction is representable.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        correction = (left[:, observable].T @ error) / scales
        remaining = right[observable].T @ correction
        # hypot avoids squaring dimensional singular values, which can
        # overflow or underflow for large or small kinematic length scales.
        if damping == 0:
            step = remaining.copy()
        else:
            attenuation = scales / np.hypot(scales, np.sqrt(damping))
            # Apply factors separately: their square may underflow while the
            # product with a large undamped correction is representable.
            step = right[observable].T @ ((correction * attenuation) * attenuation)
    if not np.all(np.isfinite(remaining)) or not np.all(np.isfinite(step)):
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
    for iteration in range(1, max_iterations + 1):
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
