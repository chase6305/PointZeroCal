"""Observability reports for ranking already-feasible contact-pose candidates."""

from dataclasses import dataclass
from math import isclose
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from ._validation import finite_scalar, positive_integer, real_array


@dataclass(frozen=True)
class ObservabilityReport:
    """SVD observability of the active offsets for one prospective data set.

    Singular values are divided by ``sqrt(matrix_rows)`` so candidate scores are
    comparable when appending one row. ``condition_number`` is infinite while
    rank is below the number of active offsets. The report describes local
    linearized information at the supplied offset seed.
    """

    active_joint_names: tuple[str, ...]
    rank: int
    singular_values: NDArray[np.float64]
    smallest_observable_singular_value: float
    condition_number: float
    matrix_rows: int

    def __post_init__(self) -> None:
        if isinstance(self.active_joint_names, (str, bytes)):
            raise ValueError("active_joint_names must be a sequence of joint names")
        try:
            names = tuple(self.active_joint_names)
        except TypeError as exc:
            raise ValueError(
                "active_joint_names must be a sequence of joint names"
            ) from exc
        values = real_array(self.singular_values, "singular_values").copy()
        rows = positive_integer(self.matrix_rows, "matrix_rows")
        if (
            not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
            or values.ndim != 1
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or values.size != min(rows, len(names))
            or np.any(values[:-1] < values[1:])
            or isinstance(self.rank, (bool, np.bool_))
            or not isinstance(self.rank, Integral)
            or not 0 <= self.rank <= len(names)
            or self.rank > np.count_nonzero(values)
        ):
            raise ValueError("invalid observability report")
        rank = int(self.rank)
        smallest = finite_scalar(
            self.smallest_observable_singular_value,
            "smallest_observable_singular_value",
            allow_zero=True,
        )
        if smallest != (float(values[rank - 1]) if rank else 0.0):
            raise ValueError("smallest observable singular value must match rank")
        condition = self.condition_number
        if (
            isinstance(condition, (bool, np.bool_))
            or not isinstance(condition, Real)
        ):
            raise ValueError("invalid observability condition number")
        try:
            condition = float(condition)
        except OverflowError as exc:
            raise ValueError("invalid observability condition number") from exc
        with np.errstate(over="ignore"):
            expected_condition = (
                float(values[0] / values[-1]) if rank == len(names) else float("inf")
            )
        if condition < 1 or not isclose(condition, expected_condition, rel_tol=1e-7):
            raise ValueError(
                "observability condition number must match the spectrum and rank"
            )
        values.setflags(write=False)
        object.__setattr__(self, "active_joint_names", names)
        object.__setattr__(self, "singular_values", values)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "matrix_rows", rows)
        object.__setattr__(self, "smallest_observable_singular_value", smallest)
        object.__setattr__(self, "condition_number", float(condition))

    @property
    def fully_observable(self) -> bool:
        """Whether every active joint has an observable local direction."""
        return self.rank == len(self.active_joint_names)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible observability diagnostics."""
        return {
            "active_joint_names": list(self.active_joint_names),
            "rank": self.rank,
            "singular_values": self.singular_values.tolist(),
            "smallest_observable_singular_value": self.smallest_observable_singular_value,
            "condition_number": (
                self.condition_number if np.isfinite(self.condition_number) else None
            ),
            "matrix_rows": self.matrix_rows,
            "fully_observable": self.fully_observable,
        }


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's local information after appending it to a contact group."""

    source_index: int
    configuration_rad: NDArray[np.float64]
    observability: ObservabilityReport
    rank_gain: int

    def __post_init__(self) -> None:
        configuration = real_array(self.configuration_rad, "configuration_rad").copy()
        if (
            isinstance(self.source_index, (bool, np.bool_))
            or not isinstance(self.source_index, (int, np.integer))
            or self.source_index < 0
            or configuration.ndim != 1
            or configuration.size == 0
            or not np.all(np.isfinite(configuration))
            or not isinstance(self.observability, ObservabilityReport)
            or isinstance(self.rank_gain, (bool, np.bool_))
            or not isinstance(self.rank_gain, Integral)
        ):
            raise ValueError("invalid candidate score")
        configuration.setflags(write=False)
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "configuration_rad", configuration)
        object.__setattr__(self, "rank_gain", int(self.rank_gain))

    def _ranking_key(self) -> tuple[int, float, float, int]:
        """Common ordering for generated and externally assembled rankings."""
        return (
            -self.observability.rank,
            -self.observability.smallest_observable_singular_value,
            self.observability.condition_number,
            self.source_index,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the candidate in radians with its prospective diagnostics."""
        return {
            "source_index": self.source_index,
            "configuration_rad": self.configuration_rad.tolist(),
            "rank_gain": self.rank_gain,
            "observability": self.observability.to_dict(),
        }


@dataclass(frozen=True)
class CandidateRanking:
    """Candidates sorted from the strongest to weakest local information gain.

    Candidates are sorted lexicographically by prospective rank, then by the
    smallest observable singular value, then by condition number. A candidate
    is only a mathematical recommendation: callers must still verify that its
    configuration is reachable and keeps the ball seated in the named socket.
    """

    group_name: str
    baseline: ObservabilityReport
    candidates: tuple[CandidateScore, ...]

    def __post_init__(self) -> None:
        scores = tuple(self.candidates)
        if (
            not isinstance(self.group_name, str)
            or not self.group_name.strip()
            or not scores
            or not isinstance(self.baseline, ObservabilityReport)
            or any(not isinstance(score, CandidateScore) for score in scores)
        ):
            raise ValueError("invalid candidate ranking")
        indices = [score.source_index for score in scores]
        if len(set(indices)) != len(indices):
            raise ValueError("candidate ranking source indices must be unique")
        for score in scores:
            if (
                score.observability.active_joint_names != self.baseline.active_joint_names
                or score.rank_gain != score.observability.rank - self.baseline.rank
            ):
                raise ValueError("candidate observability must match the ranking baseline")
        object.__setattr__(
            self, "candidates", tuple(sorted(scores, key=CandidateScore._ranking_key))
        )

    @property
    def recommended(self) -> CandidateScore:
        """Highest-ranked candidate under the documented local criterion."""
        return self.candidates[0]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible ranking in recommendation order."""
        return {
            "group_name": self.group_name,
            "baseline": self.baseline.to_dict(),
            "recommended_source_index": self.recommended.source_index,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
