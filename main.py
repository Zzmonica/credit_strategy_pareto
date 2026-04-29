"""
main.py
信贷风控多模型帕累托策略优化系统 - 主流程入口。

执行流程：
  Step 1: 生成历史多周期数据
  Step 2: 空值诊断与处理
  Step 3: 模型多样性分析
  Step 4: 帕累托优化（全量数据）
  Step 5: 跨周期稳定性验证（Walk-forward）
  Step 6: 策略筛选与产出
  Step 7: 可视化 Dashboard
  Step 8: 打印最终策略卡片
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── 业务参数配置 ─────────────────────────────────────
CONFIG = {
    # 数据生成
    "n_months":            6,
    "n_samples_per_month": 10000,
    "base_bad_rate":       0.12,
    "random_seed":         42,

    # 帕累托优化
    "n_epsilon":           35,
    "pass_rate_min":       0.40,
    "pass_rate_max":       0.88,

    # 业务目标
    "target_pass_rate":    0.70,
    "bad_rate_redline":    0.065,
    "current_pass_rate":   0.62,
    "current_bad_rate":    0.068,

    # 稳定性验证
    "walk_forward_window": 3,

    # 业务量化
    "monthly_apply":       50000,
    "avg_loan":            20000.0,
    "lgd":                 0.60,
}

SCORE_COLS = ["score_v1", "score_v2", "score_v3"]


def print_step(step: int, title: str):
    print(f"\n{'='*60}")
    print(f"  Step {step}: {title}")
    print(f"{'='*60}")


def main():
    np.random.seed(CONFIG["random_seed"])

    # ════════════════════════════════════════════════════════
    # Step 1: 生成历史多周期数据
    # ════════════════════════════════════════════════════════
    print_step(1, "生成历史多周期数据")
    from data.data_generator import generate_monthly_data

    full_df = generate_monthly_data(
        n_months             = CONFIG["n_months"],
        n_samples_per_month  = CONFIG["n_samples_per_month"],
        base_bad_rate        = CONFIG["base_bad_rate"],
        random_seed          = CONFIG["random_seed"],
    )

    print(full_df[["apply_month", "label", "gmv"] + SCORE_COLS].describe().round(4))

    # ════════════════════════════════════════════════════════
    # Step 2: 空值诊断与处理
    # ════════════════════════════════════════════════════════
    print_step(2, "空值诊断与处理")
    from src.diagnostics import diagnose_missing, handle_missing_scores

    scores_dict = {col: full_df[col].values for col in SCORE_COLS}
    labels_all  = full_df["label"].values
    gmv_all     = full_df["gmv"].values

    missing_report = diagnose_missing(scores_dict, labels_all)
    print("\n  空值诊断报告：")
    print(missing_report.to_string(index=False))

    # 处理空值
    scores_raw = full_df[SCORE_COLS].values
    scores_filled, missing_flags, need_review, weight_mask = handle_missing_scores(
        scores_raw, missing_threshold=0.30
    )

    print(f"\n  需人工审核样本: {need_review.sum()} 笔（已排除，不参与优化）")
    print(f"  有效权重掩码: {dict(zip(SCORE_COLS, weight_mask))}")

    # 排除全缺失样本
    valid_mask    = ~need_review
    scores_valid  = scores_filled[valid_mask]
    labels_valid  = labels_all[valid_mask]
    gmv_valid     = gmv_all[valid_mask]
    full_df_valid = full_df[valid_mask].reset_index(drop=True)
    # 将填充后的分数写回 DataFrame（供稳定性验证使用）
    full_df_valid = full_df_valid.copy()
    full_df_valid[SCORE_COLS] = scores_valid

    print(f"\n  有效样本量: {valid_mask.sum():,} / {len(full_df):,}")

    # ════════════════════════════════════════════════════════
    # Step 3: 模型多样性分析
    # ════════════════════════════════════════════════════════
    print_step(3, "模型多样性分析")
    from src.diagnostics import diagnose_diversity

    scores_dict_valid = {
        col: full_df_valid[col].values for col in SCORE_COLS
    }
    diversity = diagnose_diversity(scores_dict_valid, labels_valid)

    print("\n  Spearman 秩相关矩阵（越低融合收益越大）：")
    print(diversity["spearman_corr"].round(4).to_string())
    print("\n  错误相关性矩阵（越低融合收益越大）：")
    print(diversity["error_corr"].round(4).to_string())
    print("\n  十分位分桶不一致率（越高说明模型差异越大）：")
    for pair, rate in diversity["disagree_rate"].items():
        print(f"    {pair}: {rate}")

    # ════════════════════════════════════════════════════════
    # Step 4: 帕累托优化（全量数据）
    # ════════════════════════════════════════════════════════
    print_step(4, "帕累托优化（全量数据）")
    from src.pareto_optimizer import ParetoOptimizer

    optimizer = ParetoOptimizer(
        scores      = scores_valid,
        labels      = labels_valid,
        gmv         = gmv_valid,
        weight_mask = weight_mask,
    )

    candidate_df = optimizer.run_epsilon_constraint(
        n_epsilon       = CONFIG["n_epsilon"],
        pass_rate_range = (CONFIG["pass_rate_min"], CONFIG["pass_rate_max"]),
    )

    pareto_df = optimizer.filter_pareto_dominant(candidate_df)
    pareto_df["风格"] = pareto_df.apply(optimizer.label_strategy_style, axis=1)

    recommended = optimizer.select_recommended(
        pareto_df,
        target_pass_rate = CONFIG["target_pass_rate"],
        bad_rate_redline = CONFIG["bad_rate_redline"],
    )
    recommended_id = recommended["strategy_id"]

    optimal_weights = np.array([
        recommended["w_v1"],
        recommended["w_v2"],
        recommended["w_v3"],
    ])
    print(f"\n  最优权重: V1={optimal_weights[0]:.3f}, "
          f"V2={optimal_weights[1]:.3f}, V3={optimal_weights[2]:.3f}")

    # ════════════════════════════════════════════════════════
    # Step 5: 跨周期稳定性验证（Walk-forward）
    # ════════════════════════════════════════════════════════
    print_step(5, "跨周期稳定性验证（Walk-forward）")
    from src.stability_validator import StabilityValidator

    validator  = StabilityValidator(full_df_valid)

    wf_results = validator.walk_forward_validation(
        optimal_weights  = optimal_weights,
        target_pass_rate = CONFIG["target_pass_rate"],
        train_window     = CONFIG["walk_forward_window"],
    )

    psi_results = validator.calc_psi_over_time(
        optimal_weights  = optimal_weights,
        base_month_idx   = 0,
    )

    stability_summary = validator.summarize_stability(wf_results, psi_results)

    print("\n  Walk-forward 验证结果：")
    print(wf_results.to_string(index=False))
    print("\n  PSI 跨期稳定性：")
    print(psi_results.to_string(index=False))
    print("\n  稳定性摘要：")
    for k, v in stability_summary.items():
        print(f"    {k}: {v}")

    # ════════════════════════════════════════════════════════
    # Step 6: 策略筛选与产出
    # ════════════════════════════════════════════════════════
    print_step(6, "策略产出（业务量化）")
    from src.strategy_exporter import StrategyExporter

    exporter = StrategyExporter(
        monthly_apply = CONFIG["monthly_apply"],
        avg_loan      = CONFIG["avg_loan"],
        lgd           = CONFIG["lgd"],
    )

    strategy_table = exporter.export_strategy_table(
        pareto_df,
        current_pass_rate = CONFIG["current_pass_rate"],
        current_bad_rate  = CONFIG["current_bad_rate"],
    )

    display_cols = [
        "strategy_id", "风格",
        "w_v1", "w_v2", "w_v3",
        "threshold_pct", "pass_rate", "rejection_rate",
        "bad_rate", "risk_multiplier", "bad_gmv_wan",
        "auc", "ks",
        "月通过量(笔)", "月坏账量(笔)", "月坏账GMV(万元)",
        "较基准通过量变化(笔)", "较基准GMV变化(万元)",
    ]
    print("\n  Pareto 最优策略组合完整列表：")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(strategy_table[display_cols].to_string(index=False))

    strategy_table.to_csv("pareto_strategies.csv", index=False, encoding="utf-8-sig")
    print("\n  [Export] 策略表已保存至 pareto_strategies.csv")

    # ════════════════════════════════════════════════════════
    # Step 7: 可视化 Dashboard
    # ════════════════════════════════════════════════════════
    print_step(7, "可视化 Dashboard")

    exporter.plot_pareto_dashboard(
        pareto_df         = pareto_df,
        strategy_table    = strategy_table,
        wf_results        = wf_results,
        stability_summary = stability_summary,
        recommended_id    = recommended_id,
        current_pass_rate = CONFIG["current_pass_rate"],
        current_bad_rate  = CONFIG["current_bad_rate"],
        bad_rate_redline  = CONFIG["bad_rate_redline"],
        target_pass_rate  = CONFIG["target_pass_rate"],
    )

    # ════════════════════════════════════════════════════════
    # Step 8: 打印最终策略卡片
    # ════════════════════════════════════════════════════════
    print_step(8, "最终策略卡片")
    exporter.print_strategy_cards(strategy_table, n_cards=5)

    print("\n" + "★" * 68)
    print("  ★  推荐落地策略")
    print("★" * 68)
    rec_row = strategy_table[strategy_table["strategy_id"] == recommended_id]
    if not rec_row.empty:
        exporter.print_strategy_cards(rec_row, n_cards=1)

    print("\n" + "=" * 60)
    print("  全流程完成！")
    print(f"  输出文件：pareto_strategies.csv | pareto_dashboard.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
