# PointZeroCal

[简体中文](README.zh-CN.md) | English

PointZeroCal is a general-purpose joint zero-offset calibration library built on [Pinocchio](https://github.com/stack-of-tasks/pinocchio). It estimates encoder zero offsets from robot configurations in which the tool center point (TCP) touches the same fixed point.

## Installation

```bash
python -m pip install -e .
# Development and tests
python -m pip install -e '.[test]'
pytest
```

## Data requirements

Each CSV row is one joint configuration. Columns must follow the actuated-joint order in the URDF. Use widely distributed poses for observability, and keep the TCP touching the same physical point in every pose.

## Command line

```bash
point-zero-cal \
  --urdf robot.urdf \
  --end-frame tool_tip \
  --tcp-xyz 0.08 0.0 0.16 \
  --samples contact_samples.csv \
  --unit deg \
  --fixed-joint joint1
```

Results are printed to the terminal by default. Add `--output calibration.json` only when persistence is needed. `--tcp-xyz` defines the TCP translation in the end frame; `--tcp-rpy` defines its orientation in radians. Fixed-point contact cannot identify the global base rotation, so the first actuated joint is fixed by default. Repeat `--fixed-joint` to override that default. Exit code 0 means convergence; code 2 indicates non-convergence or insufficient rank.

## Python API

```python
import numpy as np
from point_zero_cal import JointZeroCalibrator, PinocchioKinematics

kinematics = PinocchioKinematics(
    "robot.urdf", "tool_tip", tcp_xyz=(0.08, 0.0, 0.16)
)
samples = np.loadtxt("contact_samples.csv", delimiter=",")
result = JointZeroCalibrator(kinematics).calibrate(
    samples, sample_unit="deg", fixed_joints=["joint1"]
)
print(result.converged, result.offsets_deg)
```

The current implementation supports fixed-base URDF models with one configuration variable per actuated joint. Floating-base and multi-DoF joints are rejected explicitly.

Results include a status message, sample count, active joints, point RMS before and after calibration, effective rank, condition number, and singular values. Diagnostics are recomputed after the final update. Input shapes, finite values, TCP parameters, and Pinocchio backend outputs are validated.

## UR10 + Viser simulation

The repository includes two comparison cases using `assets/UR10/UR10.urdf`. Both keep TCP XYZ fixed while varying RPY and use Pinocchio IK to generate configurations. The default set contains 41 poses spanning ±60°. Approximate joint ranges in the lateral-TCP case are `[35°, 47°, 44°, 115°, 96°, 147°]`.

- Axial TCP `(0, 0, 0.16)`: Joint6 is unobservable; Joint1 and Joint6 are fixed while Joint2–Joint5 are validated.
- Lateral eccentric TCP `(0.08, 0, 0.16)`: `X=0.08 m` provides a lever arm perpendicular to the Joint6 axis, making its translational Jacobian nonzero. Only Joint1 is fixed while Joint2–Joint6 are validated.

Install simulation dependencies:

```bash
python -m pip install -e '.[simulation]'
```

Run calibration and print results without writing files:

```bash
python examples/ur10_calibration_demo.py --no-viewer
```

Start Viser and use the `TCP case` dropdown and `Contact pose` slider:

```bash
python examples/ur10_calibration_demo.py
```

The pose count must be an odd integer of at least 3. Count and orientation range are configurable:

```bash
python examples/ur10_calibration_demo.py \
  --no-viewer \
  --orientation-count 49 \
  --orientation-range-deg 45
```

If the TCP lies on the Joint6 axis, fixed-point XYZ contact alone cannot identify the Joint6 offset. Use an eccentric tool or an additional orientation measurement. A relative singular-value threshold prevents floating-point noise from being mistaken for observable rank.

The demo deliberately injects Joint2–Joint6 offsets of `[+1.2°, -0.8°, +1.5°, -1.0°, +2.0°]`. The lateral-TCP case recovers all five offsets and reduces point-cloud RMS from about `4.57 mm` to numerical precision. The axial-TCP case recovers Joint2–Joint5 while retaining the unobservable Joint6 error.

Key report fields:

- `Injected`, `Estimated`, `Error`: known, recovered, and estimation-error offsets.
- `State`: whether a joint was calibrated, fixed as a reference, or unobservable.
- `Point RMS`: TCP dispersion before and after correction.
- `Rank`: number of observable joints.
- `Condition no.`: a lower value generally indicates better sample excitation.

### Optional result export

Files are written only when `--output-dir` is supplied:

```bash
python examples/ur10_calibration_demo.py \
  --no-viewer \
  --output-dir calibration_output
```

This creates `*_measured_deg.csv` simulated encoder samples and `*_result.json` calibration reports for both TCP cases.
