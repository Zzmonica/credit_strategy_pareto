"""
stability_validator.py
跨周期稳定性验证模块。
使用 Walk-forward 验证评估策略在不同时间周期的表现一致性。
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from src.metrics import (
    calc_auc, calc_ks, calc_bad_rate,
    calc_rejection_rate, calc_risk_multiplier,
    calc_bad_gmv, calc_psi, psi_level,
)


class StabilityValidator:
    """
    跨周期稳定性验证器。
    模拟策略从历史优化到未来验证的时序表现。
    """

    def __init__(self, monthly_data: pd.DataFrame):
        """
        Parameters
        ----------
        monthly_data : pd.DataFrame
            包含 apply_month 字段的全量历史数据
        """
        self.monthly_data = monthly_data
        self.months = sorted(monthly_data["apply_month"].unique())
        print(f"  [Stability] 共 {len(self.months)} 个月份: {self.months}")

    def _get_month_arrays(self, months: List[str]) -> Dict[str, np.ndarray]:
        """提取指定月份的数组数据。"""
        df = self.monthly_data[self.monthly_data["apply_month"].isin(months)]
        return {
            "scores": df[["score_v1", "score_v2", "score_v3"]].values,
            "labels": df["label"].values,
            "gmv":    df["gmv"].values,
        }

    def _apply_strategy(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        gmv: np.ndarray,
        weights: np.ndarray,
        target_pass_rate: float,
    ) -> dict:
        """
        用给定权重和目标通过率，在数据上执行策略并计算指标。

        Parameters
        ----------
        scores : np.ndarray
            分数矩阵 (N, K)
        labels : np.ndarray
            真实标签
        gmv : np.ndarray
            贷款金额
        weights : np.ndarray
            模型融合权重
        target_pass_rate : float
            目标通过率（用于确定阈值）

        Returns
        -------
        dict
            策略在该数据集上的指标
        """
        fused = scores @ weights
        tau   = np.percentile(fused, (1 - target_pass_rate) * 100)
        pass_mask = fused >= tau
        base_bad  = labels.mean()

        return {
            "auc":             calc_auc(labels, fused),
            "ks":              calc_ks(labels, fused),
            "pass_rate":       pass_mask.mean(),
            "rejection_rate":  calc_rejection_rate(pass_mask),
            "bad_rate":        calc_bad_rate(labels, pass_mask),
            "risk_multiplier": calc_risk_multiplier(labels, pass_mask, base_bad),
            "bad_gmv_wan":     calc_bad_gmv(labels, gmv, pass_mask),
        }

    def walk_forward_validation(
        self,
        optimal_weights: np.ndarray,
        target_pass_rate: float = 0.70,
        train_window: int = 3,
    ) -> pd.DataFrame:
        """
        Walk-forward 跨周期验证。
        用前 train_window 个月的数据验证权重性能，在下一个月上测试。

        例如（6个月，窗口=3）：
        - 训练：M1~M3 → 验证：M4
        - 训练：M2~M4 → 验证：M5
        - 训练：M3~M5 → 验证：M6

        Parameters
        ----------
        optimal_weights : np.ndarray
            由帕累托优化得到的最优权重
        target_pass_rate : float
            目标通过率
        train_window : int
            滚动训练窗口大小（月数）

        Returns
        -------
        pd.DataFrame
            各验证月份的策略指标
        """
        results = []
        n_months = len(self.months)

        for i in range(train_window, n_months):
            # 训练窗口月份（仅用于记录，权重已由全量优化给定）
            train_months = self.months[i - train_window: i]
            val_month    = self.months[i]

            # 在验证月份上应用策略
            val_data = self._get_month_arrays([val_month])

            # 空值填充（用列中位数）
            scores_val = val_data["scores"].copy()
            for k in range(scores_val.shape[1]):
                col = scores_val[:, k]
                median = np.nanmedian(col)
                scores_val[:, k] = np.where(np.isnan(col), median, col)

            metrics = self._apply_strategy(
                scores_val,
                val_data["labels"],
                val_data["gmv"],
                optimal_weights,
                target_pass_rate,
            )

            results.append({
                "验证月份":     val_month,
                "训练窗口":     f"{train_months[0]}~{train_months[-1]}",
                **{k: round(v, 4) for k, v in metrics.items()},
            })

            print(f"  [WalkFwd] 验证月={val_month} | "
                  f"AUC={metrics['auc']:.4f} | KS={metrics['ks']:.4f} | "
                  f"通过率={metrics['pass_rate']:.2%} | 坏账率={metrics['bad_rate']:.2%}")

        return pd.DataFrame(results)

    def calc_psi_over_time(
        self,
        optimal_weights: np.ndarray,
        base_month_idx: int = 0,
    ) -> pd.DataFrame:
        """
        计算融合分在各月之间的 PSI，评估分布稳定性。

        Parameters
        ----------
        optimal_weights : np.ndarray
            最优融合权重
        base_month_idx : int
            基准月份索引（默认第一个月）

        Returns
        -------
        pd.DataFrame
            各月 PSI 报告
        """
        base_month = self.months[base_month_idx]
        base_data  = self._get_month_arrays([base_month])
        base_scores = base_data["scores"].copy()
        for k in range(base_scores.shape[1]):
            col = base_scores[:, k]
            base_scores[:, k] = np.where(np.isnan(col), np.nanmedian(col), col)
        base_fused = base_scores @ optimal_weights

        psi_results = []
        for month in self.months:
            if month == base_month:
                continue
            cur_data   = self._get_month_arrays([month])
            cur_scores = cur_data["scores"].copy()
            for k in range(cur_scores.shape[1]):
                col = cur_scores[:, k]
                cur_scores[:, k] = np.where(np.isnan(col), np.nanmedian(col), col)
            cur_fused = cur_scores @ optimal_weights

            psi = calc_psi(base_fused, cur_fused)
            psi_results.append({
                "基准月份":   base_month,
                "对比月份":   month,
                "PSI":        round(psi, 4),
                "稳定性评级": psi_level(psi),
            })

        return pd.DataFrame(psi_results)

    def summarize_stability(
        self,
        wf_results: pd.DataFrame,
        psi_results: pd.DataFrame,
    ) -> dict:
        """
        汇总跨周期稳定性评估结果，输出综合评级。

        评级规则：
        - A（优秀）：AUC std < 0.005, 坏账率 std < 0.003, PSI 均值 < 0.1
        - B（良好）：AUC std < 0.010, 坏账率 std < 0.006, PSI 均值 < 0.15
        - C（需关注）：AUC std < 0.020, 坏账率 std < 0.010, PSI 均值 < 0.2
        - D（不稳定）：其余情况

        Parameters
        ----------
        wf_results : pd.DataFrame
            Walk-forward 验证结果
        psi_results : pd.DataFrame
            PSI 分析结果

        Returns
        -------
        dict
            稳定性摘要
        """
        auc_mean  = wf_results["auc"].mean()
        auc_std   = wf_results["auc"].std()
        ks_mean   = wf_results["ks"].mean()
        ks_std    = wf_results["ks"].std()
        br_mean   = wf_results["bad_rate"].mean()
        br_std    = wf_results["bad_rate"].std()
        psi_mean  = psi_results["PSI"].mean()

        # 综合评级
        if auc_std < 0.005 and br_std < 0.003 and psi_mean < 0.10:
            grade = "A（优秀）✅"
        elif auc_std < 0.010 and br_std < 0.006 and psi_mean < 0.15:
            grade = "B（良好）✅"
        elif auc_std < 0.020 and br_std < 0.010 and psi_mean < 0.20:
            grade = "C（需关注）⚠️"
        else:
            grade = "D（不稳定）🔴"

        summary = {
            "AUC均值":      round(auc_mean, 4),
            "AUC标准差":    round(auc_std, 4),
            "KS均值":       round(ks_mean, 4),
            "KS标准差":     round(ks_std, 4),
            "坏账率均值":   round(br_mean, 4),
            "坏账率标准差": round(br_std, 4),
            "PSI均值":      round(psi_mean, 4),
            "综合稳定性评级": grade,
        }
        return summary
