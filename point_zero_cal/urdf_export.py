"""Export calibrated URDF copies with traceable metadata.

The numeric library stays URDF-free: this module consumes calibration results
and rewrites only the selected joint ``<origin>`` attributes, writing a new
file and leaving the nominal model untouched.
"""

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ._io import validate_output_path, write_text_atomic
from ._urdf import index_joints, origin_vector, parse_joints
from ._validation import finite_real
from .geometric import PARAMETER_COMPONENTS, GeometricParameterResult

OriginUpdate = tuple[Sequence[float | None] | None, Sequence[float | None] | None]


def _format_number(value: float) -> str:
    """Format a float without scientific notation or trailing zeros."""
    return np.format_float_positional(float(value), trim="-")


def _update_attribute(
    existing: str | None, values: Sequence[float | None], attribute: str, joint: str
) -> str:
    current = origin_vector(existing, joint, attribute)
    if len(values) != 3:
        raise ValueError(
            f"origin {attribute} for joint {joint!r} must contain three values"
        )
    merged = [
        current[index] if value is None else float(value)
        for index, value in enumerate(values)
    ]
    if not np.all(np.isfinite(merged)):
        raise ValueError(f"origin {attribute} for joint {joint!r} must be finite")
    return " ".join(_format_number(value) for value in merged)


def _metadata_comment(metadata: Mapping[str, object]) -> ET.Element:
    """Encode metadata as compact JSON inside an XML comment element.

    ``--`` sequences are forbidden in XML comments and escaped as
    ``\\u002d\\u002d``, a JSON escape that decodes losslessly. ASCII JSON also
    escapes Unicode values forbidden in XML, keeping arbitrary metadata text
    from producing an unreadable URDF.
    """
    text = json.dumps(
        dict(metadata), ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).replace("--", "\\u002d\\u002d")
    return ET.Comment(f" point-zero-cal: {text} ")


def _decode_metadata(text: str) -> dict[str, object]:
    return json.loads(text)


def export_calibrated_urdf(
    source: str | Path,
    destination: str | Path,
    origin_updates: Mapping[str, OriginUpdate],
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write a calibrated copy of ``source`` to ``destination``.

    ``origin_updates`` maps joint names to ``(xyz, rpy)``, each a three-value
    sequence of absolute origin values in metres/radians with ``None`` entries
    keeping the nominal component, or ``None`` leaving the whole attribute
    untouched. Existing comments are preserved, XML formatting is normalized,
    and the source file is never modified. ``metadata`` is embedded as a
    trailing XML comment inside the robot element, so the file remains
    loadable by standard URDF parsers while carrying parameters, solver state
    and validation results.
    """
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    validate_output_path(destination_path, source_path)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(source_path, parser=parser)
    root = tree.getroot()
    joints = index_joints(root)
    for name, (xyz, rpy) in origin_updates.items():
        joint = joints.get(name)
        if joint is None:
            raise ValueError(f"joint {name!r} not found in {source_path}")
        origin = joint.find("origin")
        if origin is None:
            origin = ET.Element("origin")
            # Keep URDF's origin-first element ordering for new elements.
            joint.insert(0, origin)
        if xyz is not None:
            origin.set("xyz", _update_attribute(origin.get("xyz"), xyz, "xyz", name))
        if rpy is not None:
            origin.set("rpy", _update_attribute(origin.get("rpy"), rpy, "rpy", name))
    if metadata is not None:
        root.append(_metadata_comment(metadata))
    text = ET.tostring(
        root, encoding="unicode", xml_declaration=True, short_empty_elements=True
    )
    write_text_atomic(destination_path, text)
    return destination_path


def geometric_origin_updates(
    source: str | Path,
    parameters: Sequence[GeometricParameterResult],
) -> dict[str, OriginUpdate]:
    """Build ``origin_updates`` adding each estimate to its nominal origin.

    Estimates are deviations from the URDF values, so the exported values are
    ``nominal + estimate`` per component; components without parameters are
    kept with ``None`` entries.
    """
    source_path = Path(source).expanduser()
    urdf_joints = parse_joints(source_path)
    updates: dict[str, OriginUpdate] = {}
    seen = set()
    for parameter in parameters:
        if not isinstance(parameter, GeometricParameterResult):
            raise ValueError("parameters must contain GeometricParameterResult values")
        if parameter.component not in PARAMETER_COMPONENTS:
            raise ValueError(f"unknown geometric component: {parameter.component!r}")
        label = (parameter.joint_name, parameter.component)
        if label in seen:
            raise ValueError(f"duplicate geometric parameter: {'.'.join(label)}")
        seen.add(label)
        expected_unit = "m" if parameter.component in ("x", "y", "z") else "rad"
        if parameter.unit != expected_unit:
            raise ValueError(f"{'.'.join(label)} unit must be {expected_unit!r}")
        estimate = finite_real(parameter.estimate, f"{'.'.join(label)} estimate")
        if parameter.joint_name not in urdf_joints:
            raise ValueError(
                f"joint {parameter.joint_name!r} not found in {source_path}"
            )
        _, _, _, rpy, xyz = urdf_joints[parameter.joint_name]
        entry = updates.setdefault(parameter.joint_name, (None, None))
        xyz_values = list(entry[0]) if entry[0] is not None else [None, None, None]
        rpy_values = list(entry[1]) if entry[1] is not None else [None, None, None]
        axis = {"x": 0, "y": 1, "z": 2, "roll": 0, "pitch": 1, "yaw": 2}[
            parameter.component
        ]
        if parameter.component in ("x", "y", "z"):
            xyz_values[axis] = finite_real(
                float(xyz[axis]) + estimate, f"{'.'.join(label)} exported origin"
            )
        else:
            rpy_values[axis] = finite_real(
                float(rpy[axis]) + estimate, f"{'.'.join(label)} exported origin"
            )
        updates[parameter.joint_name] = (tuple(xyz_values), tuple(rpy_values))
    return updates
