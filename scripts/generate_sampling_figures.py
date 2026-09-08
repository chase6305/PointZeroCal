"""Generate bilingual sampling diagrams and an actual UR10 pose-coverage plot.

Run from the repository root with ``python -m scripts.generate_sampling_figures``.
Requires the optional ``docs`` dependencies. All observations are synthetic.
"""

import argparse
from html import escape
from pathlib import Path

import numpy as np

from point_zero_cal import PinocchioKinematics
from point_zero_cal.simulation import generate_fixed_point_dataset, make_rpy_samples

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, LINE = "#172b4d", "#53657d", "#dbe4ee"
BLUE, ORANGE, TEAL, PURPLE = "#2563eb", "#c26a16", "#087f8c", "#7451b8"


class Diagram:
    """Small SVG vocabulary for editable, standalone engineering diagrams."""

    def __init__(self, title: str, subtitle: str, height: int):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#53657d"/></marker></defs>',
            '<g font-family="Noto Sans CJK SC, Noto Sans CJK JP, Arial, sans-serif">',
            f'<rect width="1200" height="{height}" rx="20" fill="#f8fafc"/>',
        ]
        self.text(32, 46, title, size=28, weight=700)
        self.text(32, 78, subtitle, size=17, color=MUTED)

    def text(self, x, y, value, *, size=18, color=INK, weight=400, anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def rect(self, x, y, w, h, *, fill="white", stroke=LINE, radius=16):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'
        )

    def line(
        self,
        x1,
        y1,
        x2,
        y2,
        *,
        color=MUTED,
        width=2,
        dash=False,
        arrow=False,
        both=False,
    ):
        extra = ' stroke-dasharray="7 6"' if dash else ""
        if arrow or both:
            extra += ' marker-end="url(#arrow)"'
        if both:
            extra += ' marker-start="url(#arrow)"'
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{extra}/>'
        )

    def circle(self, x, y, r, *, fill="white", stroke=BLUE, width=2, dash=False):
        extra = ' stroke-dasharray="6 5"' if dash else ""
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{extra}/>'
        )

    def base(self, x, y, w, h, centres, r):
        path = f"M{x} {y}"
        for c in centres:
            path += f" H{c - r} C{c - r} {y + r * 0.5523} {c - r * 0.5523} {y + r} {c} {y + r} C{c + r * 0.5523} {y + r} {c + r} {y + r * 0.5523} {c + r} {y}"
        path += f" H{x + w} V{y + h} H{x} Z"
        self.parts.append(
            f'<path d="{path}" fill="#e5ecf4" stroke="#94a3b8" stroke-width="2"/>'
        )

    def save(self, path):
        path.write_text("\n".join([*self.parts, "</g></svg>"]) + "\n", encoding="utf-8")


def ball_socket(path: Path, *, zh: bool) -> None:
    def tr(cn, en):
        return cn if zh else en

    d = Diagram(
        tr(
            "球头与双球窝：如何采集同点姿态",
            "Ball and sockets: collecting fixed-point poses",
        ),
        tr(
            "理想配合的剖面示意 · TCP 定义为球心 · 图形未按比例绘制",
            "Idealized section view · TCP is the ball centre · Schematic, not to scale",
        ),
        560,
    )
    d.rect(24, 105, 552, 429)
    d.rect(596, 105, 580, 429)
    d.text(
        48,
        144,
        tr("01  固定球心，改变末端姿态", "01  Fixed centre, varied tool orientation"),
        size=22,
        weight=700,
    )
    for end, color, dashed, label in [
        ((160, 202), TEAL, True, "q1"),
        ((285, 190), BLUE, False, "q2"),
        ((411, 202), PURPLE, True, "q3"),
    ]:
        d.line(285, 330, *end, color=color, width=12, dash=dashed)
        d.text(
            end[0],
            end[1] - 20,
            label,
            color=color,
            size=20,
            weight=700,
            anchor="middle",
        )
    d.base(85, 330, 410, 92, [285], 48)
    d.circle(285, 330, 44, fill="#dbeafe", stroke=BLUE, width=3)
    d.circle(285, 330, 5, fill=BLUE, stroke="white")
    d.line(276, 330, 294, 330, color=BLUE)
    d.line(285, 321, 285, 339, color=BLUE)
    d.line(298, 330, 364, 309, color=BLUE)
    d.text(368, 309, tr("TCP = 球心", "TCP = centre"), color=BLUE, size=18, weight=700)
    d.text(
        110, 397, tr("固定球窝 / 工装", "Fixed socket / fixture"), color=MUTED, size=17
    )
    d.text(
        48,
        462,
        tr(
            "稳定落座后，记录一整行关节角。",
            "Record a full joint vector after stable seating.",
        ),
        size=18,
    )
    d.text(
        48,
        499,
        tr(
            "改变姿态，保持同一球心位置。",
            "Vary orientation while keeping the centre fixed.",
        ),
        size=17,
        color=MUTED,
    )
    d.text(
        620,
        144,
        tr("02  同一球头，依次采集两个球窝", "02  One ball, two sockets in sequence"),
        size=22,
        weight=700,
    )
    d.text(
        620,
        177,
        tr("每个球窝保存独立样本组。", "Keep one separate sample group per socket."),
        size=17,
        color=MUTED,
    )
    d.base(635, 308, 502, 82, [756, 1014], 39)
    for x, color, dashed in [(756, BLUE, False), (1014, ORANGE, True)]:
        d.line(x, 308, x, 224, color=color, width=10, dash=dashed)
        d.circle(
            x,
            308,
            35,
            fill="#eff6ff" if not dashed else "white",
            stroke=color,
            width=3,
            dash=dashed,
        )
        d.circle(x, 308, 4, fill=color, stroke="white")
        d.line(x, 354, x, 442, color=MUTED, width=1, dash=True)
    d.text(756, 373, "front/0", color=BLUE, weight=700, anchor="middle")
    d.text(1014, 373, "front/1", color=ORANGE, weight=700, anchor="middle")
    d.line(756, 431, 1014, 431, both=True)
    d.text(
        885,
        467,
        tr("d = 实测球心间距", "d = measured centre spacing"),
        size=19,
        weight=700,
        anchor="middle",
    )
    d.text(
        885,
        502,
        tr("50 mm 仅为仓库仿真示例值", "50 mm is the bundled simulation example"),
        size=17,
        color=MUTED,
        anchor="middle",
    )
    d.save(path)


def contact_workflow(path: Path, *, zh: bool) -> None:
    def tr(cn, en):
        return cn if zh else en

    d = Diagram(
        tr(
            "采样分组 → 共享零偏 → 独立验证",
            "Contact groups → shared offsets → held-out validation",
        ),
        tr(
            "工装移动后创建新组；验证数据只用于评估给定零偏。",
            "Create new groups when the fixture moves; evaluate validation data at fixed offsets.",
        ),
        460,
    )
    for x, w in [(24, 384), (448, 280), (768, 408)]:
        d.rect(x, 112, w, 280)
    d.text(48, 151, tr("位置 A · 训练", "Placement A · training"), size=22, weight=700)
    d.text(
        792, 151, tr("位置 B · 验证", "Placement B · validation"), size=22, weight=700
    )
    for x, name, color in [
        (126, "front/0", BLUE),
        (302, "front/1", ORANGE),
        (878, "left/0", TEAL),
        (1060, "left/1", PURPLE),
    ]:
        d.rect(x - 70, 185, 140, 122, fill="#f8fafc", radius=12)
        for delta in (-35, 0, 35):
            d.line(x, 238, x + delta, 207, color=color, width=3)
        d.circle(x, 238, 10, fill="white", stroke=color, width=3)
        d.text(x, 286, name, color=color, size=19, weight=700, anchor="middle")
    d.text(
        216,
        341,
        tr("组内分别计算质心", "One centroid per group"),
        size=18,
        anchor="middle",
    )
    d.text(
        216,
        371,
        tr("可选：已知球窝间距 d", "Optional: known spacing d"),
        size=16,
        color=MUTED,
        anchor="middle",
    )
    d.text(588, 176, tr("联合求解", "Joint fit"), size=22, weight=700, anchor="middle")
    d.text(588, 223, "calibrate_groups", size=19, color=BLUE, anchor="middle")
    d.text(588, 276, "δq", size=36, color=BLUE, weight=700, anchor="middle")
    d.text(
        588,
        319,
        tr("一套共享关节零偏", "One shared offset vector"),
        size=17,
        anchor="middle",
    )
    d.text(
        588,
        357,
        tr("固定参考关节，检查有效秩", "Fix reference joints; check rank"),
        size=15,
        color=MUTED,
        anchor="middle",
    )
    d.text(
        972,
        341,
        "evaluate_groups(offsets_rad=δq)",
        size=17,
        color=TEAL,
        anchor="middle",
    )
    d.text(
        972,
        371,
        tr("检查残差与间距；不更新零偏", "Check errors; keep offsets fixed"),
        size=16,
        color=MUTED,
        anchor="middle",
    )
    d.line(414, 252, 438, 252, arrow=True, width=3)
    d.line(734, 252, 758, 252, arrow=True, width=3)
    d.text(
        32,
        433,
        tr(
            "分组标识与示例导出一致：front_0_rad.csv、front_1_rad.csv、left_0_rad.csv、left_1_rad.csv",
            "Example exports: front_0_rad.csv, front_1_rad.csv, left_0_rad.csv, left_1_rad.csv",
        ),
        size=17,
        color=MUTED,
    )
    d.save(path)


def ur10_coverage(output: Path, preview: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    font = (
        font_manager.FontProperties(fname=str(font_path)).get_name()
        if font_path.is_file()
        else "sans-serif"
    )
    plt.rcParams.update(
        {
            "font.family": [font, "DejaVu Sans"],
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "path",
            "svg.hashsalt": "pzc-sampling-v1",
        }
    )
    backend = PinocchioKinematics(
        ROOT / "assets/UR10/UR10.urdf", "ee_link", tcp_xyz=(0.08, 0, 0.16)
    )
    seed = np.array([0, -1.2, 1.8, -1.4, -1.57, 0])
    origin = np.array(backend.tcp_pose(seed).translation)
    datasets = [
        generate_fixed_point_dataset(
            backend,
            seed,
            make_rpy_samples(count=21, max_angle_deg=40),
            target_xyz=origin + [0.05 * i, 0, 0],
        )
        for i in range(2)
    ]

    for zh in (False, True):

        def tr(cn, en, zh=zh):
            return cn if zh else en

        fig = plt.figure(figsize=(12, 5.8), facecolor="#f8fafc")
        fig.text(
            0.035,
            0.925,
            tr(
                "UR10：实际生成的同点采样姿态",
                "UR10: generated fixed-point sampling poses",
            ),
            fontsize=20,
            weight="bold",
        )
        fig.text(
            0.035,
            0.873,
            tr(
                "每窝 21 组姿态 · RPY 搜索范围 ±40° · 名义运动学仿真",
                "21 poses per socket · RPY search range ±40° · nominal-kinematics simulation",
            ),
            fontsize=11,
            color=MUTED,
        )
        ax = fig.add_axes(
            [0.035, 0.14, 0.47, 0.66], projection="3d", facecolor="#f8fafc"
        )
        # Show five actual IK solutions at socket 0; joint centres define a
        # kinematic skeleton, not a collision or swept-volume model.
        for i, index in enumerate((0, 3, 8, 13, 20)):
            tcp = backend.tcp_pose(datasets[0].configurations_rad[index])
            points = np.array(
                [
                    np.zeros(3),
                    *(pose.translation.copy() for pose in backend.data.oMi[1:]),
                    backend.data.oMf[backend.frame_id].translation.copy(),
                    tcp.translation.copy(),
                ]
            )
            ax.plot(
                *points.T,
                color=BLUE,
                alpha=0.22 + i * 0.13,
                linewidth=2.2,
                marker="o",
                markersize=3,
            )
        ax.scatter(
            *origin,
            s=90,
            color=ORANGE,
            edgecolors="white",
            linewidths=1.5,
            depthshade=False,
        )
        ax.text(
            *(origin + [0.03, 0, 0.08]), "TCP", color=ORANGE, fontsize=12, weight="bold"
        )
        ax.set(
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
            xlim=(-0.95, 0.12),
            ylim=(-0.43, 0.3),
            zlim=(0, 0.9),
        )
        ax.set_box_aspect((1.07, 0.73, 0.9))
        ax.view_init(elev=23, azim=-65)
        ax.tick_params(labelsize=8, pad=0)
        ax.set_title(
            tr("球窝 0：选取 5 个 IK 解", "Socket 0: five selected IK solutions"),
            fontsize=12,
            loc="left",
            pad=8,
        )
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis._axinfo["grid"]["color"] = "#e2e8f0"
        ax2 = fig.add_axes([0.59, 0.20, 0.36, 0.58], facecolor="white")
        for i, (dataset, color) in enumerate(
            zip(datasets, (BLUE, ORANGE), strict=True)
        ):
            relative = np.rad2deg(dataset.configurations_rad - seed)
            for j in range(6):
                y = j + (-0.12 if i == 0 else 0.12)
                ax2.plot(
                    [relative[:, j].min(), relative[:, j].max()],
                    [y, y],
                    color=color,
                    alpha=0.45,
                    linewidth=3,
                )
                ax2.scatter(
                    relative[:, j],
                    np.full(21, y),
                    color=color,
                    s=15,
                    alpha=0.75,
                    label=f"front/{i}" if j == 0 else None,
                )
        ax2.axvline(0, color="#cbd5e1", linewidth=1, linestyle="--")
        ax2.set_yticks(range(6), backend.joint_names)
        ax2.set_ylim(5.6, -0.6)
        ax2.set_xlabel(
            tr("相对种子姿态的关节角变化 (°)", "Joint-angle change from seed (°)"),
            labelpad=10,
        )
        ax2.set_title(
            tr("每个点代表一组采样中的该关节角", "Each dot is one sampled joint angle"),
            fontsize=12,
            loc="left",
            pad=14,
        )
        ax2.grid(axis="x", color="#e2e8f0", linewidth=0.7)
        ax2.set_axisbelow(True)
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.spines["bottom"].set_color(LINE)
        ax2.tick_params(axis="y", length=0)
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncols=2,
            frameon=False,
            fontsize=10,
        )
        fig.text(
            0.035,
            0.043,
            tr(
                "骨架与关节角来自 URDF / IK；未模拟接触力或碰撞。姿态覆盖还需结合有效秩判断。",
                "Skeleton and joint angles come from URDF / IK; contact force and collisions are not simulated. Check rank as well as coverage.",
            ),
            fontsize=9,
            color=MUTED,
        )
        suffix = ".zh-CN" if zh else ""
        fig.savefig(
            output / f"ur10-sampling{suffix}.svg",
            metadata={
                "Date": None,
                "Title": tr(
                    "UR10 同点采样姿态与关节覆盖",
                    "UR10 fixed-point poses and joint coverage",
                ),
            },
        )
        if preview:
            fig.savefig(preview / f"ur10-sampling{suffix}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--preview-dir", type=Path, help="optional PNG plot previews")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    for zh in (False, True):
        suffix = ".zh-CN" if zh else ""
        ball_socket(args.output_dir / f"ball-socket-sampling{suffix}.svg", zh=zh)
        contact_workflow(args.output_dir / f"contact-groups{suffix}.svg", zh=zh)
    ur10_coverage(args.output_dir, args.preview_dir)
    print(f"Generated six sampling figures in {args.output_dir}")


if __name__ == "__main__":
    main()
