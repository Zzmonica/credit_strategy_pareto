"""
strategy_exporter.py
策略产出与可视化模块。
将帕累托优化结果转化为业务可直接使用的策略表、Dashboard 和策略卡片。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from typing import Optional


# ══════════════════════════════════════════════════════════════
# 中文字体自动检测与注册
# 优先级：
#   1. 项目内置字体目录 fonts/
#   2. 系统已安装的中文字体（SimHei / PingFang / Noto / WenQuanYi）
#   3. 兜底：用英文替换中文标签（不乱码但变英文）
# ══════════════════════════════════════════════════════════════

def _setup_chinese_font() -> str:
    """
    自动检测可用的中文字体，返回字体名称并更新 rcParams。
    支持 Windows / macOS / Linux 三平台。

    Returns
    -------
    str
        最终生效的字体名称
    """
    # ── 候选字体优先级列表 ──
    candidate_fonts = [
        # Windows
        "SimHei", "Microsoft YaHei", "SimSun", "FangSong", "KaiTi",
        # macOS
        "PingFang SC", "Heiti SC", "STHeiti", "STSong", "STKaiti",
        # Linux（需安装 fonts-noto-cjk 或 fonts-wqy-*）
        "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei", "Droid Sans Fallback",
        # 通用
        "Arial Unicode MS", "DejaVu Sans",
    ]

    # ── Step 1: 检查项目内置字体目录 fonts/ ──
    _register_builtin_fonts()

    # ── Step 2: 在系统字体中查找可用中文字体 ──
    available = {f.name for f in fm.fontManager.ttflist}

    selected = None
    for font in candidate_fonts:
        if font in available:
            selected = font
            break

    if selected is None:
        # ── Step 3: 尝试从字体文件路径直接匹配 ──
        selected = _find_font_by_path()

    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"  [Font] 中文字体已加载: {selected}")
    else:
        # ── Step 4: 兜底 —— 全局替换为英文标签，避免乱码 ──
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print("  [Font] ⚠️  未找到中文字体，图表标签将以英文显示。")
        print("  [Font]     可安装字体解决: sudo apt-get install fonts-noto-cjk")

    return selected or "DejaVu Sans"


def _register_builtin_fonts():
    """扫描项目 fonts/ 目录，将 .ttf/.otf 文件注册到 matplotlib。"""
    fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts")
    if not os.path.isdir(fonts_dir):
        return
    for fname in os.listdir(fonts_dir):
        if fname.lower().endswith((".ttf", ".otf")):
            fpath = os.path.join(fonts_dir, fname)
            fm.fontManager.addfont(fpath)
    # 重建字体缓存
    fm._load_fontmanager(try_read_cache=False)


def _find_font_by_path() -> Optional[str]:
    """
    直接扫描系统字体路径，寻找包含 CJK 字符的字体文件，
    注册后返回字体名称。
    """
    cjk_keywords = ["cjk", "chinese", "noto", "wqy", "pingfang",
                     "simhei", "yahei", "heiti", "songti", "wenquanyi"]
    for font in fm.fontManager.ttflist:
        fname_lower = os.path.basename(font.fname).lower()
        if any(kw in fname_lower for kw in cjk_keywords):
            try:
                fm.fontManager.addfont(font.fname)
                return font.name
            except Exception:
                continue
    return None


# ── 模块加载时自动配置字体 ──
_FONT_NAME = _setup_chinese_font()


def _cn(zh: str, en: str) -> str:
    """
    根据当前字体是否支持中文，返回中文或英文标签。
    避免使用不支持中文的字体时出现方块乱码。
    """
    if _FONT_NAME == "DejaVu Sans":
        return en
    return zh


# ══════════════════════════════════════════════════════════════
# StrategyExporter 主类
# ══════════════════════════════════════════════════════════════

class StrategyExporter:
    """
    策略产出器：业务量化、可视化 Dashboard、策略卡片打印。
    """

    def __init__(
        self,
        monthly_apply: int = 50000,
        avg_loan: float = 20000.0,
        lgd: float = 0.6,
    ):
        """
        Parameters
        ----------
        monthly_apply : int
            月申请量（笔）
        avg_loan : float
            平均贷款金额（元）
        lgd : float
            违约损失率（Loss Given Default）
        """
        self.monthly_apply = monthly_apply
        self.avg_loan      = avg_loan
        self.lgd           = lgd

    def export_strategy_table(
        self,
        pareto_df: pd.DataFrame,
        current_pass_rate: float = 0.62,
        current_bad_rate: float = 0.068,
    ) -> pd.DataFrame:
        """
        在 Pareto 策略表基础上补充月度业务量化指标，并与当前策略对比。

        Parameters
        ----------
        pareto_df : pd.DataFrame
            Pareto 最优策略表
        current_pass_rate : float
            当前策略通过率（基准）
        current_bad_rate : float
            当前策略坏账率（基准）

        Returns
        -------
        pd.DataFrame
            补充业务指标后的完整策略表
        """
        df = pareto_df.copy()
        df["风格"] = df.apply(self._label_style, axis=1)

        # 月度量化指标
        df["月通过量(笔)"]    = (df["pass_rate"] * self.monthly_apply).astype(int)
        df["月坏账量(笔)"]    = (df["pass_rate"] * self.monthly_apply * df["bad_rate"]).astype(int)
        df["月坏账GMV(万元)"] = (
            df["pass_rate"] * self.monthly_apply * df["bad_rate"] * self.avg_loan * self.lgd / 1e4
        ).round(1)

        # 当前策略基准
        cur_pass_cnt = current_pass_rate * self.monthly_apply
        cur_bad_gmv  = cur_pass_cnt * current_bad_rate * self.avg_loan * self.lgd / 1e4

        df["较基准通过量变化(笔)"] = df["月通过量(笔)"] - int(cur_pass_cnt)
        df["较基准GMV变化(万元)"]  = (df["月坏账GMV(万元)"] - cur_bad_gmv).round(1)

        return df

    @staticmethod
    def _label_style(row: pd.Series) -> str:
        """根据通过率打业务风格标签。"""
        pr = row["pass_rate"]
        if pr < 0.50:   return "极度保守"
        elif pr < 0.60: return "保守型"
        elif pr < 0.72: return "平衡型"
        elif pr < 0.82: return "激进型"
        else:           return "极度激进"

    def plot_pareto_dashboard(
        self,
        pareto_df: pd.DataFrame,
        strategy_table: pd.DataFrame,
        wf_results: pd.DataFrame,
        stability_summary: dict,
        recommended_id: str,
        current_pass_rate: float = 0.62,
        current_bad_rate: float = 0.068,
        bad_rate_redline: float = 0.065,
        target_pass_rate: float = 0.70,
    ) -> plt.Figure:
        """
        绘制 2x2 业务决策 Dashboard。

        四张子图：
        - 左上：Pareto 前沿散点图（含标注）
        - 右上：模型权重演变折线图
        - 左下：Walk-forward 跨周期 AUC/KS 折线图
        - 右下：各策略月坏账 GMV 柱状图
        """
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor("#F8F9FA")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

        ax1 = fig.add_subplot(gs[0, 0])   # 左上：Pareto 前沿
        ax2 = fig.add_subplot(gs[0, 1])   # 右上：权重演变
        ax3 = fig.add_subplot(gs[1, 0])   # 左下：跨周期稳定性
        ax4 = fig.add_subplot(gs[1, 1])   # 右下：坏账 GMV 对比

        self._plot_pareto_frontier(
            ax1, pareto_df, recommended_id,
            current_pass_rate, current_bad_rate,
            bad_rate_redline, target_pass_rate,
        )
        self._plot_weight_evolution(ax2, pareto_df)
        self._plot_walkforward(ax3, wf_results, stability_summary)
        self._plot_bad_gmv_bars(ax4, strategy_table, current_pass_rate, current_bad_rate)

        fig.suptitle(
            _cn("信贷风控多模型帕累托策略优化 Dashboard",
                "Credit Risk Multi-Model Pareto Strategy Dashboard"),
            fontsize=18, fontweight="bold", y=1.01,
            color="#1A237E",
        )
        plt.savefig("pareto_dashboard.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("\n  [Export] Dashboard 已保存至 pareto_dashboard.png")
        return fig

    # ── 子图1：Pareto 前沿 ──────────────────────────────
    def _plot_pareto_frontier(
        self, ax, pareto_df, recommended_id,
        current_pass_rate, current_bad_rate,
        bad_rate_redline, target_pass_rate,
    ):
        ax.set_facecolor("#FAFAFA")

        sc = ax.scatter(
            pareto_df["pass_rate"],
            pareto_df["bad_rate"],
            c=pareto_df["bad_rate"],
            cmap="RdYlGn_r",
            s=100, zorder=5,
            edgecolors="white", linewidth=0.8,
            vmin=pareto_df["bad_rate"].min(),
            vmax=pareto_df["bad_rate"].max(),
        )
        ax.plot(
            pareto_df["pass_rate"],
            pareto_df["bad_rate"],
            "k--", lw=1.2, alpha=0.35, zorder=4,
        )
        plt.colorbar(sc, ax=ax,
                     label=_cn("坏账率", "Bad Rate"),
                     format="%.2f", shrink=0.85)

        # 推荐策略高亮
        rec = pareto_df[pareto_df["strategy_id"] == recommended_id]
        if not rec.empty:
            rec = rec.iloc[0]
            ax.scatter(rec["pass_rate"], rec["bad_rate"],
                       s=300, c="#4CAF50", marker="*", zorder=10,
                       edgecolors="white", linewidth=1.5,
                       label=_cn("推荐策略", "Recommended"))
            ax.annotate(
                (f"  {_cn('推荐', 'Rec')}: {recommended_id}\n"
                 f"  {_cn('通过率', 'PassRate')} {rec['pass_rate']:.1%}\n"
                 f"  {_cn('坏账率', 'BadRate')} {rec['bad_rate']:.2%}"),
                xy=(rec["pass_rate"], rec["bad_rate"]),
                xytext=(rec["pass_rate"] + 0.03, rec["bad_rate"] + 0.003),
                fontsize=8.5, color="#2E7D32", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#4CAF50", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#4CAF50"),
            )

        # 当前策略位置
        ax.scatter(current_pass_rate, current_bad_rate,
                   s=200, c="black", marker="X", zorder=10,
                   label=_cn("当前策略", "Current"))
        ax.annotate(
            (f"  {_cn('当前策略', 'Current')}\n"
             f"  {_cn('通过率', 'Pass')} {current_pass_rate:.0%}\n"
             f"  {_cn('坏账率', 'Bad')} {current_bad_rate:.1%}"),
            xy=(current_pass_rate, current_bad_rate),
            xytext=(current_pass_rate - 0.12, current_bad_rate + 0.005),
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        ax.axvline(target_pass_rate, color="#1565C0", lw=1.5, ls=":",
                   label=_cn(f"目标通过率 {target_pass_rate:.0%}",
                              f"Target PassRate {target_pass_rate:.0%}"))
        ax.axhline(bad_rate_redline, color="#B71C1C", lw=1.5, ls=":",
                   label=_cn(f"风控红线 {bad_rate_redline:.1%}",
                              f"RedLine {bad_rate_redline:.1%}"))

        ax.fill_between(
            [target_pass_rate, pareto_df["pass_rate"].max() + 0.05],
            [0, 0],
            [bad_rate_redline, bad_rate_redline],
            alpha=0.07, color="#4CAF50",
        )
        ax.text(target_pass_rate + 0.01, bad_rate_redline * 0.45,
                _cn("✓ 业务可行域", "✓ Feasible Zone"),
                fontsize=8.5, color="#2E7D32", fontweight="bold")

        ax.set_xlabel(_cn("通过率  →  越高规模越大", "Pass Rate  →  Larger Scale"),
                      fontsize=11)
        ax.set_ylabel(_cn("通过人群坏账率  →  越低风险越低", "Bad Rate  →  Lower Risk"),
                      fontsize=11)
        ax.set_title(_cn("Pareto 效率前沿", "Pareto Efficiency Frontier"),
                     fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(True, alpha=0.2)

    # ── 子图2：权重演变 ─────────────────────────────────
    def _plot_weight_evolution(self, ax, pareto_df):
        ax.set_facecolor("#FAFAFA")
        colors = {"w_v1": "#2196F3", "w_v2": "#FF9800", "w_v3": "#4CAF50"}
        labels = {
            "w_v1": _cn("V1 权重", "V1 Weight"),
            "w_v2": _cn("V2 权重", "V2 Weight"),
            "w_v3": _cn("V3 权重", "V3 Weight"),
        }

        for col, color in colors.items():
            ax.plot(pareto_df["pass_rate"], pareto_df[col],
                    "o-", color=color, lw=2, ms=4, label=labels[col])
            ax.fill_between(pareto_df["pass_rate"], pareto_df[col],
                            alpha=0.1, color=color)

        ax.set_xlabel(_cn("通过率（策略激进程度）→", "Pass Rate (Aggressiveness) →"),
                      fontsize=11)
        ax.set_ylabel(_cn("模型权重", "Model Weight"), fontsize=11)
        ax.set_title(_cn("各模型权重随策略激进程度的变化",
                         "Model Weight vs Strategy Aggressiveness"),
                     fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.text(0.02, 0.92,
                _cn("← 保守策略更依赖高精度模型\n→ 激进策略权重更分散",
                    "<- Conservative relies on high-AUC model\n-> Aggressive spreads weights"),
                transform=ax.transAxes, fontsize=8, color="gray",
                bbox=dict(boxstyle="round", fc="white", ec="lightgray", alpha=0.8))

    # ── 子图3：Walk-forward 稳定性 ──────────────────────
    def _plot_walkforward(self, ax, wf_results, stability_summary):
        ax.set_facecolor("#FAFAFA")
        months = wf_results["验证月份"].astype(str)
        x = range(len(months))

        ax.plot(x, wf_results["auc"], "o-", color="#1565C0", lw=2,
                ms=6, label="AUC", zorder=5)
        ax.axhline(stability_summary["AUC均值"], color="#1565C0",
                   lw=1, ls="--", alpha=0.5)

        ax2 = ax.twinx()
        ax2.plot(x, wf_results["ks"], "s--", color="#E65100", lw=2,
                 ms=6, label="KS", zorder=5)
        ax2.axhline(stability_summary["KS均值"], color="#E65100",
                    lw=1, ls=":", alpha=0.5)
        ax2.set_ylabel("KS", fontsize=10, color="#E65100")
        ax2.tick_params(axis="y", labelcolor="#E65100")

        auc_mean = stability_summary["AUC均值"]
        auc_std  = stability_summary["AUC标准差"]
        ax.fill_between(x,
                        [auc_mean - auc_std] * len(x),
                        [auc_mean + auc_std] * len(x),
                        alpha=0.12, color="#1565C0",
                        label="AUC \u00b11\u03c3")

        ax.set_xticks(list(x))
        ax.set_xticklabels(months, rotation=30, fontsize=8)
        ax.set_ylabel("AUC", fontsize=10, color="#1565C0")

        grade = stability_summary["综合稳定性评级"]
        ax.set_title(
            (_cn("Walk-forward 跨周期稳定性验证", "Walk-forward Stability Validation")
             + f"\n{_cn('综合评级', 'Grade')}: {grade}  |  "
             + f"AUC={auc_mean:.4f}(\u00b1{auc_std:.4f})"),
            fontsize=11, fontweight="bold",
        )
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.2)

    # ── 子图4：坏账 GMV 对比 ────────────────────────────
    def _plot_bad_gmv_bars(self, ax, strategy_table, current_pass_rate, current_bad_rate):
        ax.set_facecolor("#FAFAFA")

        n = min(6, len(strategy_table))
        indices = np.linspace(0, len(strategy_table) - 1, n, dtype=int)
        sample  = strategy_table.iloc[indices].reset_index(drop=True)

        cur_bad_gmv = (
            current_pass_rate * self.monthly_apply * current_bad_rate
            * self.avg_loan * self.lgd / 1e4
        )

        bar_colors = [
            "#E53935" if v > cur_bad_gmv else "#43A047"
            for v in sample["月坏账GMV(万元)"]
        ]
        bars = ax.bar(
            sample["strategy_id"],
            sample["月坏账GMV(万元)"],
            color=bar_colors, alpha=0.85,
            edgecolor="white", linewidth=1.2,
        )
        ax.axhline(cur_bad_gmv, color="black", lw=2, ls="--",
                   label=_cn(f"当前策略 ¥{cur_bad_gmv:.0f}万",
                              f"Current ¥{cur_bad_gmv:.0f}wan"))

        for bar, (_, row) in zip(bars, sample.iterrows()):
            delta = row["月坏账GMV(万元)"] - cur_bad_gmv
            sign  = "+" if delta >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + cur_bad_gmv * 0.01,
                    f"\u00a5{row['月坏账GMV(万元)']:.0f}\n({sign}{delta:.0f})",
                    ha="center", va="bottom", fontsize=8,
                    color="#B71C1C" if delta > 0 else "#2E7D32",
                    fontweight="bold")

        ax.set_xlabel(_cn("策略 ID", "Strategy ID"), fontsize=11)
        ax.set_ylabel(_cn("月坏账 GMV（万元）", "Monthly Bad GMV (wan)"), fontsize=11)
        ax.set_title(
            _cn("各策略月坏账 GMV 对比（红=高于当前，绿=低于当前）",
                "Monthly Bad GMV by Strategy (Red=Higher, Green=Lower)"),
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)

        for bar, (_, row) in zip(bars, sample.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    _cn(f"通过率\n{row['pass_rate']:.0%}",
                        f"Pass\n{row['pass_rate']:.0%}"),
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")

    def print_strategy_cards(
        self,
        strategy_table: pd.DataFrame,
        n_cards: int = 5,
    ) -> None:
        """
        打印代表性策略卡片（均匀覆盖保守～激进范围）。

        Parameters
        ----------
        strategy_table : pd.DataFrame
            补充业务指标后的完整策略表
        n_cards : int
            打印策略数量
        """
        indices  = np.linspace(0, len(strategy_table) - 1, n_cards, dtype=int)
        selected = strategy_table.iloc[indices].reset_index(drop=True)

        print("\n" + "=" * 68)
        print("         信贷风控策略候选方案（帕累托最优组合）")
        print("=" * 68)

        for _, row in selected.iterrows():
            pr  = row["pass_rate"]
            br  = row["bad_rate"]
            rm  = row["risk_multiplier"]
            rej = row["rejection_rate"]

            print(f"""
┌─────────────────────────────────────────────────────────────┐
│  策略 {row['strategy_id']}  {row['风格']}
├─────────────────────────────────────────────────────────────┤
│  模型权重：V1={row['w_v1']:.3f}  V2={row['w_v2']:.3f}  V3={row['w_v3']:.3f}
│  拦截阈值：{row['threshold_pct']}（融合分 >= {row['threshold']:.4f} 通过）
├──────────────────────┬──────────────────────────────────────┤
│  通过率：  {pr:.1%}       │  拒绝率：  {rej:.1%}
│  坏账率：  {br:.2%}      │  风险倍率：{rm:.2f}x
│  AUC：     {row['auc']:.4f}      │  KS：      {row['ks']:.4f}
│  Gini：    {row['gini']:.4f}      │  坏账GMV：¥{row['bad_gmv_wan']:.0f}万/月
├──────────────────────┴──────────────────────────────────────┤
│  月通过量：{row['月通过量(笔)']:,} 笔  ({row['较基准通过量变化(笔)']:+,} vs 当前)
│  月坏账量：{row['月坏账量(笔)']:,} 笔
│  月坏账GMV：¥{row['月坏账GMV(万元)']:.0f} 万  ({row['较基准GMV变化(万元)']:+.0f}万 vs 当前)
└─────────────────────────────────────────────────────────────┘""")
