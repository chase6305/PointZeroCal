"""Shared validation for numerical solver and simulation options."""

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def kinematics_metadata(dof: int, joint_names: Sequence[str]) -> tuple[str, ...]:
    """Require a positive DoF and one distinct, non-empty name per joint."""
    count = positive_integer(dof, "kinematics.dof")
    message = "kinematics.joint_names must contain one unique name per DoF"
    if isinstance(joint_names, (str, bytes)):
        raise ValueError(message)
    try:
        names = tuple(joint_names)
    except TypeError as exc:
        raise ValueError(message) from exc
    if (
        len(names) != count
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != count
    ):
        raise ValueError(message)
    return names


def positive_integer(value: int, name: str) -> int:
    """Require an integer greater than zero, excluding booleans."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def finite_scalar(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Require a finite positive (or non-negative) real scalar."""
    bound = "non-negative" if allow_zero else "positive"
    message = f"{name} must be finite and {bound}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)  # noqa: TRY004 - shared public validation contract
    try:
        scalar = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not np.isfinite(scalar) or (scalar < 0 if allow_zero else scalar <= 0):
        raise ValueError(message)
    return scalar


def finite_real(value: float, name: str) -> float:
    """Require any finite real scalar, including zero and negative values."""
    message = f"{name} must be finite"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)  # noqa: TRY004 - shared public validation contract
    try:
        scalar = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not np.isfinite(scalar):
        raise ValueError(message)
    return scalar


def real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert numeric arrays without silently discarding imaginary values."""
    if isinstance(value, np.ndarray) and value.dtype == np.float64:
        return value
    message = f"{name} must contain real numeric values"
    try:
        array = np.asarray(value)
        if array.dtype.kind == "c":
            raise ValueError(message)
        return np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
