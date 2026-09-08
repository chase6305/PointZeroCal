"""Shared URDF origin parsing for geometric evaluation and export."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def index_joints(root: ET.Element) -> dict[str, ET.Element]:
    """Index unambiguous joint names without silently overwriting duplicates."""
    if root.tag != "robot":
        raise ValueError("URDF root element must be 'robot'")
    joints = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name or not name.strip():
            raise ValueError("URDF joint must have a non-empty name")
        if name in joints:
            raise ValueError(f"duplicate URDF joint name: {name!r}")
        joints[name] = joint
    return joints


def origin_vector(value: str | None, joint: str, attribute: str) -> NDArray[np.float64]:
    """Read exactly three finite metre/radian components; missing means zero."""
    if value is None:
        return np.zeros(3)
    message = f"joint {joint!r}: malformed origin {attribute}; expected three finite values"
    try:
        values = np.array([float(part) for part in value.split()])
    except ValueError as exc:
        raise ValueError(message) from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(message)
    return values


def parse_joints(
    path: Path,
) -> dict[str, tuple[str, str, str, NDArray, NDArray]]:
    """Return name -> (type, parent link, child link, origin rpy, origin xyz)."""
    joints = {}
    for name, joint in index_joints(ET.parse(path).getroot()).items():
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        joints[name] = (
            joint.get("type", "unknown"),
            parent.get("link", "") if parent is not None else "",
            child.get("link", "") if child is not None else "",
            origin_vector(origin.get("rpy") if origin is not None else None, name, "rpy"),
            origin_vector(origin.get("xyz") if origin is not None else None, name, "xyz"),
        )
    return joints
