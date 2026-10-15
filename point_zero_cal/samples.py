"""Read joint samples while preserving column identity and source row numbers."""

import csv
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._validation import real_array


@dataclass(frozen=True)
class SampleTable:
    """CSV configurations in model joint order, without unit conversion.

    ``values`` has shape (N, dof). ``csv_rows`` contains the one-based physical
    source line number of each sample, including gaps for headers and comments.
    """

    values: NDArray[np.float64]
    joint_names: tuple[str, ...]
    csv_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        names = _joint_names(self.joint_names)
        values = real_array(self.values, "samples").copy()
        rows = tuple(self.csv_rows)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(names):
            raise ValueError(f"samples must have shape (N, {len(names)}) with N >= 2")
        if not np.all(np.isfinite(values)):
            raise ValueError("samples must contain only finite values")
        if (
            len(rows) != len(values)
            or any(
                isinstance(row, (bool, np.bool_))
                or not isinstance(row, (int, np.integer))
                or row < 1
                for row in rows
            )
            or any(a >= b for a, b in pairwise(rows))
        ):
            raise ValueError(
                "csv_rows must contain one increasing positive integer per sample"
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "csv_rows", tuple(int(row) for row in rows))


def _joint_names(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(  # noqa: TRY004 - public input validation uses ValueError
            "joint_names must be a sequence of unique joint names"
        )
    try:
        names = tuple(value)
    except TypeError as exc:
        raise ValueError(
            "joint_names must be a sequence of unique joint names"
        ) from exc
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("joint_names must contain unique non-empty strings")
    return names


def _without_comment(line: str) -> str:
    """Retain hashes inside quoted CSV fields; ignore comments outside quotes."""
    if "#" not in line:
        return line
    quoted = False
    for index, character in enumerate(line):
        if character == '"':
            quoted = not quoted
        elif character == "#" and not quoted:
            return line[:index]
    return line


def _column_order(cells: list[str], names: tuple[str, ...], location: str) -> list[int]:
    if any(not cell for cell in cells):
        raise ValueError(f"{location}: CSV header contains an empty joint name")
    duplicates = [name for name, count in Counter(cells).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"{location}: duplicate joint names in CSV header: {duplicates}"
        )
    missing = [name for name in names if name not in cells]
    unknown = [name for name in cells if name not in names]
    if missing or unknown:
        raise ValueError(
            f"{location}: CSV header does not match model joints; "
            f"missing joint names: {missing}; unknown joint names: {unknown}"
        )
    return [cells.index(name) for name in names]


def load_samples_csv(
    path: str | Path,
    joint_names: Sequence[str],
    *,
    header: bool = False,
) -> SampleTable:
    """Read one configuration per physical line, optionally matching a header.

    With ``header=True``, the first non-empty, non-comment row must name every
    model joint exactly once. Columns are reordered to ``joint_names``. The
    default reads numeric rows in model order and never guesses a header.
    UTF-8 BOMs, quoted cells, blank lines and unquoted ``#`` comments are accepted.
    Errors identify the physical line and, for invalid values, source column.
    """
    names = _joint_names(joint_names)
    if not isinstance(header, (bool, np.bool_)):
        raise ValueError("header must be a boolean")  # noqa: TRY004
    path = Path(path).expanduser()
    order = list(range(len(names)))
    source_names = names
    need_header = bool(header)
    rows: list[list[float]] = []
    line_numbers: list[int] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = _without_comment(line).strip()
                if not line:
                    continue
                location = f"{path}: line {line_number}"
                try:
                    cells = [
                        cell.strip()
                        for cell in next(
                            csv.reader([line], strict=True, skipinitialspace=True)
                        )
                    ]
                except csv.Error as exc:
                    raise ValueError(f"{location}: invalid CSV: {exc}") from exc
                if need_header:
                    order = _column_order(cells, names, location)
                    source_names = tuple(cells)
                    need_header = False
                    continue
                if len(cells) != len(names):
                    raise ValueError(
                        f"{location}: expected {len(names)} joint columns, got {len(cells)}"
                    )
                row = []
                for column, cell in enumerate(cells, start=1):
                    try:
                        value = float(cell)
                    except ValueError as exc:
                        raise ValueError(
                            f"{location}, column {column} ({source_names[column - 1]!r}): "
                            f"cannot convert {cell!r} to a number"
                        ) from exc
                    if not math.isfinite(value):
                        raise ValueError(
                            f"{location}, column {column} ({source_names[column - 1]!r}): "
                            "samples must contain only finite values"
                        )
                    row.append(value)
                rows.append([row[index] for index in order])
                line_numbers.append(line_number)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"sample CSV not found: {path}") from exc
    except UnicodeError as exc:
        raise ValueError(f"{path}: sample CSV must be UTF-8 encoded") from exc
    if len(rows) < 2:
        raise ValueError(
            f"{path}: samples must have shape (N, {len(names)}) with N >= 2; "
            f"found {len(rows)} sample rows"
        )
    return SampleTable(np.asarray(rows, dtype=float), names, tuple(line_numbers))
