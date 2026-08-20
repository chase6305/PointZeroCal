# PointZeroCal

简体中文 | [English](README.md)

PointZeroCal 是基于 [Pinocchio](https://github.com/stack-of-tasks/pinocchio) 的通用机器人关节零位偏差标定库。它利用多组“工具中心点（TCP）接触同一固定点”时的关节角，通过末端平移雅可比迭代估计编码器零偏。

## 安装

```bash
python -m pip install -e .
# 开发与测试
python -m pip install -e '.[test]'
pytest
```

## 数据要求

CSV 每行是一组关节角，列顺序必须与 URDF 的可动关节顺序一致。实际标定应采用分布充分的姿态来保证可观性，并确保所有姿态下 TCP 都接触同一个物理点。

## 命令行

```bash
point-zero-cal \
  --urdf robot.urdf \
  --end-frame tool_tip \
  --tcp-xyz 0.08 0.0 0.16 \
  --samples contact_samples.csv \
  --unit deg \
  --fixed-joint joint1
```

结果默认打印到终端，不写入文件；需要保存时添加 `--output calibration.json`。`--tcp-xyz` 指定 TCP 在末端 frame 下的平移，`--tcp-rpy` 指定其弧度制姿态。同点接触无法辨识整体基座旋转，因此默认固定第一个可动关节。可重复使用 `--fixed-joint` 覆盖默认值。返回码 0 表示收敛，2 表示未收敛或有效秩不足。

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

当前支持固定基座、每个可动关节一个配置变量的 URDF 模型。浮动基座及多自由度关节会明确报错。

标定结果包含状态说明、样本数、活动关节、标定前后点云 RMS、有效秩、条件数和奇异值。算法在最终更新后重新计算诊断量，并校验输入形状、非有限数值、TCP 参数和 Pinocchio 后端输出。

## UR10 + Viser 仿真验证

仓库提供基于 `assets/UR10/UR10.urdf` 的两组对照案例。两者都固定 TCP 的 XYZ、改变 RPY，并通过 Pinocchio IK 生成姿态。默认使用 41 个 ±60° 范围内的分散姿态；横向 TCP 案例的六关节跨度约为 `[35°, 47°, 44°, 115°, 96°, 147°]`。

- 轴向 TCP `(0, 0, 0.16)`：Joint6 不可观；固定 Joint1/Joint6，验证 Joint2–Joint5。
- 横向偏心 TCP `(0.08, 0, 0.16)`：`X=0.08 m` 是垂直于第六轴的力臂，使 Joint6 的平移雅可比非零；仅固定 Joint1，验证 Joint2–Joint6。

安装仿真依赖：

```bash
python -m pip install -e '.[simulation]'
```

只运行标定并打印结果，不保存文件：

```bash
python examples/ur10_calibration_demo.py --no-viewer
```

启动 Viser，通过 `TCP case` 下拉框和 `Contact pose` 滑块查看案例：

```bash
python examples/ur10_calibration_demo.py
```

点位数必须是大于等于 3 的奇数，可调整数量和姿态范围：

```bash
python examples/ur10_calibration_demo.py \
  --no-viewer \
  --orientation-count 49 \
  --orientation-range-deg 45
```

如果 TCP 位于 Joint6 轴线上，仅靠固定点 XYZ 无法辨识 Joint6，需要横向偏心工具或额外姿态测量。标定器使用相对奇异值阈值，避免将浮点噪声误认为有效可观秩。

示例故意向 Joint2–Joint6 注入 `[+1.2°, -0.8°, +1.5°, -1.0°, +2.0°]` 零偏。横向 TCP 案例可以恢复全部五项，并将点云 RMS 从约 `4.57 mm` 降至数值精度；轴向 TCP 案例恢复 Joint2–Joint5，同时保留不可观的 Joint6 误差。

报告中的关键字段：

- `Injected`、`Estimated`、`Error`：注入值、估计值和估计误差。
- `State`：关节是参与标定、固定为参考，还是不可观。
- `Point RMS`：修正前后的 TCP 点云离散程度。
- `Rank`：可观关节数量。
- `Condition no.`：通常越小代表样本激励越好。

### 可选：导出结果

只有显式提供 `--output-dir` 才会写文件：

```bash
python examples/ur10_calibration_demo.py \
  --no-viewer \
  --output-dir calibration_output
```

输出包括两种案例的 `*_measured_deg.csv` 模拟编码器采集值，以及 `*_result.json` 标定报告。
