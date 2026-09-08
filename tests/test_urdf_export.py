"""Calibrated URDF export: preservation, rewriting and metadata roundtrips."""

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from point_zero_cal import PinocchioKinematics
from point_zero_cal.geometric import GeometricParameter, GeometricParameterResult
from point_zero_cal.geometric_backend import PinocchioGeometricKinematics
from point_zero_cal.urdf_export import (
    _decode_metadata,
    export_calibrated_urdf,
    geometric_origin_updates,
)

UR10 = Path(__file__).resolve().parents[1] / "assets/UR10/UR10.urdf"

ANNOTATED = """<?xml version="1.0"?>
<robot name="planar">
  <!-- a comment inside the robot element -->
  <link name="base"/>
  <link name="link1"/>
  <link name="tool"/>
  <link name="tip_link"/>
  <!-- between links -->
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <origin xyz="1 0 0"/>
    <parent link="link1"/><child link="tool"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="tip" type="fixed">
    <parent link="tool"/><child link="tip_link"/><origin xyz="1 0 0"/>
  </joint>
</robot>
"""


@pytest.mark.parametrize(
    "source_text",
    [
        ANNOTATED.replace('name="joint1"', 'name="joint2"'),
        ANNOTATED.replace('<robot name="planar">', '<model>').replace(
            '</robot>', '</model>'
        ),
    ],
)
def test_ambiguous_or_nonrobot_source_preserves_destination(tmp_path, source_text):
    source, destination = tmp_path / "source.urdf", tmp_path / "output.urdf"
    source.write_text(source_text)
    destination.write_text("previous result")
    with pytest.raises(ValueError, match="duplicate|root element"):
        export_calibrated_urdf(source, destination, {"joint2": ([1, 0, 0], None)})
    assert destination.read_text() == "previous result"


@pytest.mark.parametrize("attribute", ["xyz", "rpy"])
@pytest.mark.parametrize("value", ["1 2", "1 2 3 4", "nan 0 0", "1 x 0"])
def test_geometric_paths_reject_malformed_origins(tmp_path, attribute, value):
    source = tmp_path / "source.urdf"
    source.write_text(ANNOTATED.replace(
        '<origin xyz="1 0 0"/>', f'<origin {attribute}="{value}"/>'
    ))
    parameter = GeometricParameterResult(
        "joint2", "z", "m", 0, 0.01, 1, True, None, None
    )
    with pytest.raises(ValueError, match=f"joint2.*origin {attribute}"):
        geometric_origin_updates(source, [parameter])
    with pytest.raises(ValueError, match=f"joint2.*origin {attribute}"):
        PinocchioGeometricKinematics(
            source, "tip_link", [GeometricParameter("joint2", "z", 1)]
        )


@pytest.mark.parametrize("change, message", [
    ({"component": "invalid"}, "component"),
    ({"unit": "rad"}, "unit"),
    ({"estimate": float("nan")}, "finite"),
    ({"estimate": float("inf")}, "finite"),
])
def test_export_updates_reject_invalid_parameter_results(change, message):
    parameter = GeometricParameterResult(
        "Joint2", "y", "m", 0, 0.01, 1, True, None, None
    )
    with pytest.raises(ValueError, match=message):
        geometric_origin_updates(UR10, [replace(parameter, **change)])


def test_export_updates_reject_duplicate_parameters():
    parameter = GeometricParameterResult(
        "Joint2", "y", "m", 0, 0.01, 1, True, None, None
    )
    with pytest.raises(ValueError, match="duplicate geometric parameter"):
        geometric_origin_updates(UR10, [parameter, replace(parameter, estimate=0.02)])


def origin_attributes(path, joint_name):
    root = ET.parse(path).getroot()
    joint = next(j for j in root.findall("joint") if j.get("name") == joint_name)
    origin = joint.find("origin")
    return origin.get("xyz"), origin.get("rpy")


def test_export_rewrites_only_requested_components(tmp_path):
    source = tmp_path / "nominal.urdf"
    destination = tmp_path / "calibrated.urdf"
    source.write_text(ANNOTATED, encoding="utf-8")
    export_calibrated_urdf(
        source,
        destination,
        {"joint2": ([1.5, None, 0.2], None)},
        metadata={"solver": "ok"},
    )
    xyz, rpy = origin_attributes(destination, "joint2")
    assert xyz == "1.5 0 0.2"
    assert rpy is None
    # Untouched joints keep their nominal origins.
    assert origin_attributes(destination, "tip")[0] == "1 0 0"
    # Comments inside the robot element survive the rewrite; document-level
    # comments outside the root element are not preserved by ElementTree.
    text = destination.read_text(encoding="utf-8")
    assert "a comment inside the robot element" in text
    assert "between links" in text
    # The source file is byte-identical.
    assert source.read_bytes() == ANNOTATED.encode()


def test_export_creates_missing_origin_and_keeps_nominal_components(tmp_path):
    source = tmp_path / "nominal.urdf"
    destination = tmp_path / "calibrated.urdf"
    source.write_text(
        ANNOTATED.replace('    <origin xyz="1 0 0"/>\n', ""), encoding="utf-8"
    )
    export_calibrated_urdf(source, destination, {"joint2": ([None, 0.7, None], None)})
    # A joint without an origin element has nominal zero components.
    assert origin_attributes(destination, "joint2")[0] == "0 0.7 0"
    PinocchioKinematics(destination, "tip_link")  # must stay loadable


def test_exported_urdf_matches_the_deviated_model(tmp_path):
    parameters = [
        GeometricParameter("Joint2", "y", 0.01),
        GeometricParameter("Joint2", "pitch", 0.01),
        GeometricParameter("jointEE", "z", 0.01),
    ]
    backend = PinocchioGeometricKinematics(
        UR10, "ee_link", parameters, tcp_xyz=(0.08, 0, 0.16)
    )
    deviations = np.array([0.003, -0.02, 0.004])
    q = np.array([0.1, -1.0, 1.5, -1.2, 0.7, 0.3])
    deviated_position, _ = backend.position_and_geometric_jacobian(q, deviations)
    destination = tmp_path / "calibrated.urdf"
    # Updates are absolute origin values (nominal + deviation); build them
    # through the helper so the nominal rpy/xyz values stay consistent.
    updates = geometric_origin_updates(
        UR10,
        [
            GeometricParameterResult(
                "Joint2", "y", "m", 0.0, 0.003, 0.01, True, None, None
            ),
            GeometricParameterResult(
                "Joint2", "pitch", "rad", 0.0, -0.02, 0.01, True, None, None
            ),
            GeometricParameterResult(
                "jointEE", "z", "m", 0.0, 0.004, 0.01, True, None, None
            ),
        ],
    )
    export_calibrated_urdf(UR10, destination, updates)
    rebuilt = PinocchioKinematics(destination, "ee_link", tcp_xyz=(0.08, 0, 0.16))
    np.testing.assert_allclose(
        rebuilt.position_and_jacobian(q)[0], deviated_position, atol=1e-12
    )


def test_geometric_origin_updates_adds_estimates_to_nominal(tmp_path):
    destination = tmp_path / "calibrated.urdf"
    parameters = [
        GeometricParameterResult(
            "Joint2", "y", "m", 0.0, 0.003, 0.01, True, None, None
        ),
        GeometricParameterResult(
            "jointEE", "roll", "rad", 0.0, 0.015, 0.01, True, None, None
        ),
    ]
    updates = geometric_origin_updates(UR10, parameters)
    export_calibrated_urdf(UR10, destination, updates)
    root = ET.parse(destination).getroot()
    joint2 = next(j for j in root.findall("joint") if j.get("name") == "Joint2")
    origin = joint2.find("origin")
    assert origin.get("xyz") == "0 0.223941 0"
    joint_ee = next(j for j in root.findall("joint") if j.get("name") == "jointEE")
    rpy = joint_ee.find("origin").get("rpy")
    np.testing.assert_allclose(
        np.asarray([float(v) for v in rpy.split()]),
        [-1.57079632679 + 0.015, 0.0, 0.0],
        atol=1e-12,
    )


def test_metadata_comment_round_trips_with_double_dash_escaping(tmp_path):
    source = tmp_path / "nominal.urdf"
    destination = tmp_path / "calibrated.urdf"
    source.write_text(ANNOTATED, encoding="utf-8")
    metadata = {
        "message": "geometric calibration converged",
        "note": "a--b--c",
        "literal_escape": r"\u002d\u002d",
        "backslash_dash": "\\--",
        "nested": {"rank": 2, "tolerance": 1e-6},
    }
    export_calibrated_urdf(
        source, destination, {"joint2": ([1.5, None, None], None)}, metadata=metadata
    )
    text = destination.read_text(encoding="utf-8")
    # The comment content itself must never contain a double dash.
    start = text.index("point-zero-cal: ")
    end = text.index("-->", start)
    assert "--" not in text[start:end]
    decoded = _decode_metadata(text[start + len("point-zero-cal: ") : end])
    assert decoded == json.loads(json.dumps(metadata))
    assert decoded["note"] == "a--b--c"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metadata_preserves_existing_destination(tmp_path, value):
    source = tmp_path / "nominal.urdf"
    destination = tmp_path / "calibrated.urdf"
    source.write_text(ANNOTATED, encoding="utf-8")
    destination.write_text("existing model", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        export_calibrated_urdf(source, destination, {}, metadata={"value": value})
    assert destination.read_text(encoding="utf-8") == "existing model"


@pytest.mark.parametrize("note", ["标定完成 🦾", "\ufffe", "\uffff", "\ud800"])
def test_unicode_metadata_remains_parseable_xml_and_round_trips(tmp_path, note):
    source = tmp_path / "source.urdf"
    destination = tmp_path / "result.urdf"
    source.write_text(ANNOTATED, encoding="utf-8")
    export_calibrated_urdf(source, destination, {}, metadata={"note": note})
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(destination, parser=parser).getroot()
    comments = [element.text for element in root if element.tag is ET.Comment]
    metadata = next(text for text in comments if "point-zero-cal:" in text)
    assert _decode_metadata(metadata.split("point-zero-cal:", 1)[1]) == {"note": note}


def test_export_rejects_source_aliasing_and_unknown_joints(tmp_path):
    source = tmp_path / "nominal.urdf"
    source.write_text(ANNOTATED, encoding="utf-8")
    with pytest.raises(ValueError, match="output must differ"):
        export_calibrated_urdf(source, source, {})
    with pytest.raises(ValueError, match="not found"):
        export_calibrated_urdf(source, tmp_path / "out.urdf", {"missing": (None, None)})
    with pytest.raises(ValueError, match="three values"):
        export_calibrated_urdf(
            source, tmp_path / "out.urdf", {"joint2": ([1.0, 2.0], None)}
        )


def test_export_rejects_malformed_existing_origins(tmp_path):
    source = tmp_path / "broken.urdf"
    destination = tmp_path / "out.urdf"
    source.write_text(
        ANNOTATED.replace('<origin xyz="1 0 0"/>', '<origin xyz="1 0"/>'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed"):
        export_calibrated_urdf(
            source, destination, {"joint2": ([None, 0.5, None], None)}
        )
