"""Named contact observations and fixed-parameter evaluation reports."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite_scalar, real_array
from .samples import SampleTable


@dataclass(frozen=True)
class ContactGroup:
    """Joint configurations touching one stationary point.

    ``samples`` has shape (N, dof), N >= 2. Units are selected when solving or
    evaluating. Use a new ``name`` whenever the physical contact point moves.
    Optional ``csv_rows`` are one-based physical source lines. Optional
    ``sigma_m`` is the one-sigma positional noise of this group's samples in
    metres; ``None`` keeps the default equal weighting across groups.
    """

    name: str
    samples: ArrayLike
    csv_rows: tuple[int, ...] | None = None
    sigma_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("contact group name must be a non-empty string")
        if self.sigma_m is not None:
            object.__setattr__(self, "sigma_m", finite_scalar(self.sigma_m, "sigma_m"))
        values = real_array(self.samples, "group samples")
        if values.ndim != 2 or min(values.shape) < 1 or len(values) < 2:
            raise ValueError("group samples must have shape (N, dof), N >= 2")
        # Share the source-row validation and detached, read-only array contract.
        rows = self.csv_rows
        table = SampleTable(
            values,
            tuple(str(i) for i in range(values.shape[1])),
            tuple(range(1, len(values) + 1)) if rows is None else rows,
        )
        object.__setattr__(self, "samples", table.values)
        if rows is not None:
            object.__setattr__(self, "csv_rows", table.csv_rows)


@dataclass(frozen=True)
class KnownDistance:
    """Measured distance in metres between two named contact-group centres.

    ``weight`` is a positive dimensionless squared-residual weight. Acceptance
    still checks the unweighted distance error against ``distance_tolerance``.
    Optional ``sigma_m`` is the one-sigma measurement uncertainty of the
    distance in metres; ``None`` leaves the residual at its ``weight`` scaling.
    """

    first: str
    second: str
    distance_m: float
    weight: float = 1.0
    sigma_m: float | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(name, str) or not name.strip()
            for name in (self.first, self.second)
        ):
            raise ValueError("distance endpoints must be non-empty group names")
        if self.first == self.second:
            raise ValueError("distance endpoints must name different contact groups")
        for name in ("distance_m", "weight"):
            object.__setattr__(self, name, finite_scalar(getattr(self, name), name))
        if self.sigma_m is not None:
            object.__setattr__(self, "sigma_m", finite_scalar(self.sigma_m, "sigma_m"))


@dataclass(frozen=True)
class GroupEvaluation:
    """One group's centroid and Euclidean errors in metres, in sample order."""

    name: str
    contact_point_xyz_m: NDArray[np.float64]
    point_errors_m: NDArray[np.float64]
    point_rms_m: float
    csv_rows: tuple[int, ...] | None = None
    sigma_m: float | None = None

    def __post_init__(self) -> None:
        for name in ("contact_point_xyz_m", "point_errors_m"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def worst_sample_index(self) -> int:
        """Zero-based index of the largest residual in this group."""
        return int(np.argmax(self.point_errors_m))

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-compatible diagnostics."""
        result: dict[str, object] = {
            "name": self.name,
            "sample_count": len(self.point_errors_m),
            "contact_point_xyz_m": self.contact_point_xyz_m.tolist(),
            "point_errors_m": self.point_errors_m.tolist(),
            "point_rms_m": self.point_rms_m,
            "max_point_error_m": float(np.max(self.point_errors_m)),
            "worst_sample_index": self.worst_sample_index,
            "sigma_m": self.sigma_m,
        }
        if self.csv_rows is not None:
            result["sample_csv_rows"] = list(self.csv_rows)
            result["worst_sample_csv_row"] = self.csv_rows[self.worst_sample_index]
        return result


@dataclass(frozen=True)
class DistanceEvaluation:
    """Signed error is predicted minus measured centre distance, in metres."""

    first: str
    second: str
    distance_m: float
    predicted_distance_m: float
    error_m: float
    weight: float
    sigma_m: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "first": self.first,
            "second": self.second,
            "distance_m": self.distance_m,
            "predicted_distance_m": self.predicted_distance_m,
            "error_m": self.error_m,
            "weight": self.weight,
            "sigma_m": self.sigma_m,
        }


@dataclass(frozen=True)
class ContactEvaluation:
    """Evaluate fixed offsets; group centroids are inferred, offsets are not fit.

    ``point_rms_m`` equally weights contact groups. ``residual_rms_m`` includes
    the weighted squared distance errors, without dilution by sample count.
    These metrics describe contact consistency and relative distances, not
    absolute TCP accuracy against an independently measured world position.
    """

    offsets_rad: NDArray[np.float64]
    groups: tuple[GroupEvaluation, ...]
    distances: tuple[DistanceEvaluation, ...]
    point_rms_m: float
    residual_rms_m: float

    def __post_init__(self) -> None:
        offsets = np.array(self.offsets_rad, dtype=float, copy=True)
        offsets.setflags(write=False)
        object.__setattr__(self, "offsets_rad", offsets)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible evaluation data with explicit units."""
        return {
            "offsets_rad": self.offsets_rad.tolist(),
            "groups": [group.to_dict() for group in self.groups],
            "distances": [distance.to_dict() for distance in self.distances],
            "point_rms_m": self.point_rms_m,
            "residual_rms_m": self.residual_rms_m,
        }
