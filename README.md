# PointZeroCal

[简体中文](README.zh-CN.md) | English

Estimate robot joint zero offsets from poses recorded while the tool centre point (TCP) touches a fixed point. Built on Pinocchio, with a backend-independent Python solver and a JSON command-line interface.

Supports multiple contact groups, known socket distances, geometric URDF-origin calibration, and validation on independent observations. Reports rank, singular values, condition number, and per-sample residuals.

## Install and try

Requires Python 3.10+. Run from the repository root; installation includes NumPy and Pinocchio.

```bash
python -m pip install -e .
python examples/ur10_calibration_demo.py --no-viewer
python examples/ur10_two_socket_demo.py
```

The UR10 examples compare axial and eccentric TCPs, then fit two sockets with encoder noise and validate at an independent fixture placement. The default two-socket simulation reduces held-out RMS from about **3.40 mm to 0.036 mm**.

For the interactive viewer: `python -m pip install -e '.[simulation]'`, then run `python examples/ur10_calibration_demo.py`.

## Collect data

![Sample varied orientations in each stationary socket, then validate at a separate fixture placement](docs/assets/ball-socket-sampling.svg)

1. Set the TCP to the centre of the ball attached to the tool.
2. Seat the ball in one stationary socket and record complete joint configurations at varied orientations.
3. Save a separate CSV for each socket or fixture placement; optionally measure centre spacing.
4. Hold out an independent placement and evaluate it using the fitted parameters.

Each CSV row contains all actuated joint angles. Columns follow URDF joint order, or use a joint-name header with `--csv-header`. Fixed joints remain in the CSV and may move during sampling; only their estimated offsets are fixed.

## Command line

Use your robot model and a CSV with a joint-name header:

```bash
point-zero-cal --urdf robot.urdf --end-frame tool_tip \
  --samples samples.csv --csv-header --unit deg \
  --tcp-xyz 0.08 0 0.16 --residual-tolerance 0.0001
```

Results are printed as JSON. Add `--output calibration.json` to save them, and omit `--csv-header` for headerless CSVs. The first actuated joint is fixed by default; repeat `--fixed-joint NAME` to choose reference joints. Exit code 0 indicates convergence and successful validation, when supplied; code 2 indicates failure or invalid input.

For multiple contacts and independent validation:

```bash
point-zero-cal --urdf robot.urdf --end-frame tool_tip --csv-header --unit rad \
  --tcp-xyz 0.08 0 0.16 \
  --group front0=front_0_rad.csv --group front1=front_1_rad.csv \
  --distance front0 front1 0.05 \
  --validate-group left0=left_0_rad.csv --validate-group left1=left_1_rad.csv \
  --validate-distance left0 left1 0.05
```

Replace `0.05` with the measured spacing in metres. Run `point-zero-cal --help` for noise weights, tolerances, and other options.

## Python API

```python
import numpy as np
from point_zero_cal import JointZeroCalibrator, PinocchioKinematics, load_samples_csv

model = PinocchioKinematics("robot.urdf", "tool_tip", tcp_xyz=(0.08, 0, 0.16))
table = load_samples_csv("samples.csv", model.joint_names, header=True)
fit = JointZeroCalibrator(model).calibrate(table.values, sample_unit="deg")
if not fit.converged:
    raise RuntimeError(fit.message)
corrected_rad = np.deg2rad(table.values) + fit.offsets_rad
print(fit.offsets_deg, fit.corrected_point_rms)
```

CLI input defaults to degrees; Python input defaults to radians. TCP translations, distances, and residual tolerances use metres. Apply offsets by adding `offsets_rad` to measured angles in radians.

Use `calibrate_groups` for named contacts and distances, `evaluate_groups` for fixed-parameter validation, and `GeometricCalibrator` for selected origin deviations. See the [usage guide](docs/usage.md) for complete examples, custom backends, uncertainty weighting, and URDF export.

## Scope and validation

- The Pinocchio adapter supports fixed-base models with bounded, single-DoF revolute joints.
- Fixed-point measurements retain a global base-rotation ambiguity. A TCP on the wrist axis also leaves that wrist offset unobservable; inspect rank and pose coverage.
- Held-out RMS measures contact consistency and relative distances. Check independent observations to assess the fitted model; the examples report simulation results.

```bash
python -m pip install -e '.[test]'
pytest
python -m compileall point_zero_cal tests
```

[Usage guide](docs/usage.md) · [Numerical and performance notes](docs/optimization-notes.md) · [Design review](docs/kinematics_calibration_design_review.md) · [Figure sources](docs/assets/README.md)
