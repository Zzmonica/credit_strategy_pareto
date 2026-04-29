"""
pareto_optimizer.py
帕累托优化核心模块。
使用 ε-约束法枚举 Pareto 前沿，每个点对应一套完整策略参数。
不依赖模型重训练，仅使用已有模型的预测分数。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
from src.metrics import (
    calc_auc, calc_ks, calc_gini,
    calc_bad_rate, calc_rejection_rate,
    calc_risk_multiplier, calc_bad_gmv,
)


class ParetoOptimizer:
    """
    基于 ε-约束法的多模型融合帕累托策略优化器。

    核心思路：
    固定通过率约束（ε），在该约束下最小化坏账率，
    扫描 ε 的取值范围，描绘完整的 Pareto 效率前沿。
    """

    def __init__(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        gmv: np.ndarray,
        weight_mask: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        scores : np.ndarray
            填充后的分数矩阵，shape (N, K)，无 NaN
        labels : np.ndarray
            真实标签，shape (N,)
        gmv : np.ndarray
            贷款金额，shape (N,)
        weight_mask : np.ndarray, optional
            权重掩码 (K,)，0 表示该模型被排除（空值率过高）
        """
        self.scores     = scores
        self.labels     = labels
        self.gmv        = gmv
        self.N, self.K  = scores.shape
        self.base_bad_rate = labels.mean()
        self.weight_mask   = weight_mask if weight_mask is not None else np.ones(self.K)

        print(f"  [Pareto] 初始化完成 | 样本量={self.N:,} | "
              f"模型数={self.K} | 整体坏账率={self.base_bad_rate:.2%}")

    def _fused_score(self, w: np.ndarray) -> np.ndarray:
        """计算融合分数（权重归一化后的加权平均）。"""
        w = np.abs(w) * self.weight_mask          # 应用权重掩码
        w_sum = w.sum()
        if w_sum < 1e-9:
            return self.scores.mean(axis=1)
        w = w / w_sum                             # 归一化到和为1
        return self.scores @ w

    def _compute_strategy_metrics(
        self,
        w: np.ndarray,
        rejection_rate: float,
    ) -> dict:
        """
        给定权重和拒绝率，计算完整的策略指标。

        Parameters
        ----------
        w : np.ndarray
            模型权重（归一化前）
        rejection_rate : float
            拒绝率（即分位数阈值的百分位）

        Returns
        -------
        dict
            包含所有策略指标的字典
        """
        fused = self._fused_score(w)
        tau   = np.percentile(fused, rejection_rate * 100)   # 拒绝率对应分位数
        pass_mask = fused >= tau

        # 归一化权重（用于记录）
        w_norm = np.abs(w) * self.weight_mask
        w_norm = w_norm / (w_norm.sum() + 1e-9)

        return {
            "w":               w_norm,
            "threshold":       tau,
            "pass_rate":       pass_mask.mean(),
            "rejection_rate":  calc_rejection_rate(pass_mask),
            "bad_rate":        calc_bad_rate(self.labels, pass_mask),
            "risk_multiplier": calc_risk_multiplier(self.labels, pass_mask, self.base_bad_rate),
            "bad_gmv":         calc_bad_gmv(self.labels, self.gmv, pass_mask),
            "auc":             calc_auc(self.labels, fused),
            "ks":              calc_ks(self.labels, fused),
            "gini":            calc_gini(self.labels, fused),
        }

    def run_epsilon_constraint(
        self,
        n_epsilon: int = 40,
        pass_rate_range: tuple = (0.40, 0.90),
    ) -> pd.DataFrame:
        """
        用 ε-约束法枚举 Pareto 前沿上的策略。

        对每个目标通过率 ε，求解：
            min  bad_rate(w)
            s.t. pass_rate(w) >= ε
                 sum(w) = 1, w >= 0

        Parameters
        ----------
        n_epsilon : int
            ε 扫描点数量
        pass_rate_range : tuple
            通过率扫描范围 (min, max)

        Returns
        -------
        pd.DataFrame
            所有策略点的完整参数表
        """
        strategy_list = []
        pass_rates = np.linspace(pass_rate_range[0], pass_rate_range[1], n_epsilon)

        print(f"  [Pareto] 正在枚举 {n_epsilon} 个 ε 约束点...")

        for idx, target_pr in enumerate(pass_rates):
            rejection_rate = 1 - target_pr   # 对应拒绝率（分位数阈值）

            # ── 目标函数：最小化坏账率 ──
            def objective(w, rej=rejection_rate):
                fused = self._fused_score(w)
                tau   = np.percentile(fused, rej * 100)
                mask  = fused >= tau
                if mask.sum() < 30:
                    return 1.0
                return float(self.labels[mask].mean())

            # ── 通过率约束：实际通过率 >= target_pr ──
            def pass_rate_con(w, tpr=target_pr, rej=rejection_rate):
                fused = self._fused_score(w)
                tau   = np.percentile(fused, rej * 100)
                return float((fused >= tau).mean()) - tpr

            # 多次随机初始化，取最优结果（避免局部最优）
            best_result = None
            for trial in range(5):
                w0 = np.random.dirichlet(np.ones(self.K))   # Dirichlet 初始化
                res = minimize(
                    objective,
                    w0,
                    method="SLSQP",
                    bounds=[(0, 1)] * self.K,
                    constraints=[
                        {"type": "eq",   "fun": lambda w: w.sum() - 1},
                        {"type": "ineq", "fun": pass_rate_con},
                    ],
                    options={"ftol": 1e-9, "maxiter": 500},
                )
                if res.success and (best_result is None or res.fun < best_result.fun):
                    best_result = res

            if best_result is None or not best_result.success:
                continue

            # ── 计算完整策略指标 ──
            metrics = self._compute_strategy_metrics(best_result.x, rejection_rate)
            if np.isnan(metrics["bad_rate"]):
                continue

            w_norm = metrics["w"]
            strategy_list.append({
                "strategy_id":    f"S{idx+1:02d}",
                "w_v1":           round(w_norm[0], 3),
                "w_v2":           round(w_norm[1], 3),
                "w_v3":           round(w_norm[2], 3),
                "threshold":      round(metrics["threshold"], 4),
                "threshold_pct":  f"Top {target_pr*100:.0f}%",
                "pass_rate":      round(metrics["pass_rate"], 4),
                "rejection_rate": round(metrics["rejection_rate"], 4),
                "bad_rate":       round(metrics["bad_rate"], 4),
                "risk_multiplier":round(metrics["risk_multiplier"], 3),
                "bad_gmv_wan":    round(metrics["bad_gmv"], 1),
                "auc":            round(metrics["auc"], 4),
                "ks":             round(metrics["ks"], 4),
                "gini":           round(metrics["gini"], 4),
            })

        df = pd.DataFrame(strategy_list)
        print(f"  [Pareto] 枚举完成，候选策略 {len(df)} 个")
        return df

    def filter_pareto_dominant(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从候选策略中筛选真正的非支配解（Pareto 最优前沿）。

        非支配定义：不存在另一策略在通过率更高且坏账率更低。
        即：按通过率降序排列，依次保留坏账率单调递减的解。

        Parameters
        ----------
        df : pd.DataFrame
            候选策略表

        Returns
        -------
        pd.DataFrame
            Pareto 最优策略表
        """
        df_sorted = df.sort_values("pass_rate", ascending=False).reset_index(drop=True)
        pareto_idx = []
        min_bad = np.inf

        for i, row in df_sorted.iterrows():
            if row["bad_rate"] < min_bad:
                pareto_idx.append(i)
                min_bad = row["bad_rate"]

        pareto_df = df_sorted.loc[pareto_idx].sort_values("pass_rate").reset_index(drop=True)
        print(f"  [Pareto] 非支配筛选后，Pareto 前沿策略 {len(pareto_df)} 个")
        return pareto_df

    @staticmethod
    def label_strategy_style(row: pd.Series) -> str:
        """
        根据通过率为策略打业务风格标签。

        Parameters
        ----------
        row : pd.Series
            策略行

        Returns
        -------
        str
            风格标签
        """
        pr = row["pass_rate"]
        if pr < 0.50:
            return "🛡️ 极度保守"
        elif pr < 0.60:
            return "🔵 保守型"
        elif pr < 0.72:
            return "✅ 平衡型"
        elif pr < 0.82:
            return "🟠 激进型"
        else:
            return "🔴 极度激进"

    def select_recommended(
        self,
        pareto_df: pd.DataFrame,
        target_pass_rate: float = 0.70,
        bad_rate_redline: float = 0.065,
    ) -> pd.Series:
        """
        在 Pareto 前沿上选择推荐策略。
        规则：在坏账率 <= 红线的前提下，选通过率最接近目标的策略。

        Parameters
        ----------
        pareto_df : pd.DataFrame
            Pareto 最优策略表
        target_pass_rate : float
            业务目标通过率
        bad_rate_redline : float
            风控红线坏账率

        Returns
        -------
        pd.Series
            推荐策略行
        """
        feasible = pareto_df[pareto_df["bad_rate"] <= bad_rate_redline]
        if len(feasible) == 0:
            print("  [Pareto] ⚠️ 无满足红线约束的策略，放宽约束取坏账率最低点")
            feasible = pareto_df

        diff = (feasible["pass_rate"] - target_pass_rate).abs()
        best = feasible.loc[diff.idxmin()]
        print(f"  [Pareto] 推荐策略：{best['strategy_id']} | "
              f"通过率={best['pass_rate']:.1%} | 坏账率={best['bad_rate']:.2%} | "
              f"风险倍率={best['risk_multiplier']:.2f}")
        return best
