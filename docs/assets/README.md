# Sampling figures / 采样配图

The sampling figures are generated from repository-owned vector drawing code and UR10 kinematics. They do not contain third-party photographs or hardware measurement data.

采样配图来自仓库中的矢量绘图代码和 UR10 运动学计算，不包含第三方照片或实物测量数据。中文文件使用 `.zh-CN.svg` 后缀。

| Figure / 配图 | Contents / 内容 |
| --- | --- |
| [ball-socket-sampling.svg](ball-socket-sampling.svg) / [中文](ball-socket-sampling.zh-CN.svg) | Idealized ball/socket section, fixed ball-centre TCP, sequential use of two sockets, measured centre spacing. / 球头与球窝理想剖面、固定球心 TCP、双球窝依次采样和实测中心间距。 |
| [contact-groups.svg](contact-groups.svg) / [中文](contact-groups.zh-CN.svg) | Separate contact groups, shared offset fitting, evaluation at another placement with fixed parameters. / 独立接触组、共享零偏拟合、另一工装位置的固定参数验证。 |
| [ur10-sampling.svg](ur10-sampling.svg) / [中文](ur10-sampling.zh-CN.svg) | Five actual IK skeletons at one socket and all 21 joint configurations at each of two sockets. / 同一球窝的五个实际 IK 骨架解，以及两个球窝各 21 组关节配置。 |

## Regenerate / 重新生成

Run from the repository root / 从仓库根目录运行：

```bash
python -m pip install -e '.[docs]'
python -m scripts.generate_sampling_figures
```

The generator writes six standalone SVGs. It uses the bundled `assets/UR10/UR10.urdf`, seed configuration `[0, -1.2, 1.8, -1.4, -1.57, 0] rad`, eccentric TCP `[0.08, 0, 0.16] m`, 21 symmetric RPY samples with a ±40° search bound, and socket centres separated by 0.05 m along world X. The skeleton view selects sample indices `[0, 3, 8, 13, 20]`. Joint-angle changes are measured relative to the seed configuration. These are nominal IK configurations before adding the offset and noise used by the calibration demo.

生成器写入六张独立 SVG。模型、种子姿态、偏心 TCP、每窝 21 组姿态及 50 mm 间距与双球窝示例一致；骨架图展示索引 `[0, 3, 8, 13, 20]` 的解，关节角变化相对种子姿态计算。图中使用注入零偏和噪声之前的名义 IK 配置。骨架连线不代表碰撞几何，仿真不模拟接触力。

For Chinese plots, the generator uses Noto Sans CJK at `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` when available. Install this font or configure an equivalent Matplotlib CJK font before regeneration on another platform. Plot text is embedded as vector paths, so viewing the exported plot does not require the font. The two hand-drawn diagrams retain editable SVG text with system-font fallbacks.

中文统计图在当前环境中使用 Noto Sans CJK，文字导出为矢量轮廓；查看成图不需安装该字体。在其他平台重新生成时，需准备相应中文字体。两张原理示意图保留可编辑 SVG 文字并配置系统字体回退。

Optional plot previews / 可选生成统计图 PNG 预览：

```bash
python -m scripts.generate_sampling_figures --preview-dir /tmp/pzc-sampling-preview
```

SVG timestamps are omitted and path identifiers use a fixed hash salt. Regeneration is deterministic within the same NumPy, Pinocchio, Matplotlib and font environment; different versions may change layout or floating-point output. The original overview PNGs are retained and are not rewritten by this generator.
