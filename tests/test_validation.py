"""Shared backend metadata validation across the two calibration APIs."""

from types import SimpleNamespace

import pytest

from point_zero_cal import GeometricCalibrator, GeometricParameter, JointZeroCalibrator


@pytest.mark.parametrize("calibrator", [JointZeroCalibrator, GeometricCalibrator])
@pytest.mark.parametrize("dof, names", [
    (0, ()),
    (True, ("joint",)),
    (2.0, ("a", "b")),
    (2, "ab"),
    (2, None),
    (2, ("a",)),
    (2, ("a", "a")),
    (2, ("a", " ")),
    (2, ("a", ["b"])),
])
def test_invalid_metadata_fails_before_numerical_evaluation(calibrator, dof, names):
    backend = SimpleNamespace(
        dof=dof,
        joint_names=names,
        geometric_parameters=(GeometricParameter("a", "x", 1.0),),
    )
    with pytest.raises(ValueError, match="kinematics"):
        calibrator(backend)
