"""
diagnostics.py
模型分诊断模块：空值分析、多样性分析、空值处理。
在进入帕累托优化前完成数据质量评估与预处理。
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def diagnose_missing(
    scores_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    诊断各模型分的空值情况，并检验空值与坏账标签的相关性。

    Parameters
    ----------
    scores_dict : dict
        {'v1': array, 'v2': array, 'v3': array}
    labels : np.ndarray
        真实标签

    Returns
    -------
    pd.DataFrame
        空值诊断报告
    """
    rows = []
    for name, score in scores_dict.items():
        missing_mask = np.isnan(score)
        n_missing = missing_mask.sum()
        missing_rate = missing_mask.mean()

        # 空值样本坏账率
        bad_rate_missing = (
            labels[missing_mask].mean() if n_missing > 0 else np.nan
        )
        # 非空样本坏账率
        bad_rate_valid = (
            labels[~missing_mask].mean() if (~missing_mask).sum() > 0 else np.nan
        )

        # 卡方检验：空值是否与坏账显著相关
        p_value = np.nan
        is_mnar = False
        if n_missing > 10:
            ct = pd.crosstab(missing_mask, labels)
            if ct.shape == (2, 2):
                _, p_value, _, _ = stats.chi2_contingency(ct)
                is_mnar = p_value < 0.05   # 显著相关 → MNAR（空值有含义）

        rows.append({
            "模型":          name,
            "空值数量":      n_missing,
            "空值率":        f"{missing_rate:.2%}",
            "空值样本坏账率": f"{bad_rate_missing:.2%}" if not np.isnan(bad_rate_missing) else "—",
            "非空样本坏账率": f"{bad_rate_valid:.2%}",
            "相关性p值":     f"{p_value:.4f}" if not np.isnan(p_value) else "—",
            "缺失类型":      "MNAR（有含义）⚠️" if is_mnar else "MAR/MCAR（随机）✅",
            "空值率原始":    missing_rate,   # 供后续处理逻辑使用
        })

    return pd.DataFrame(rows)


def diagnose_diversity(
    scores_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    decision_threshold: float = 0.5,
) -> dict:
    """
    分析多个模型分之间的差异性（多样性）。
    多样性越高，融合收益越大。

    Parameters
    ----------
    scores_dict : dict
        各模型分数字典
    labels : np.ndarray
        真实标签
    decision_threshold : float
        用于计算分类误差的阈值

    Returns
    -------
    dict
        包含相关矩阵、错误相关矩阵、不一致率的诊断结果
    """
    names = list(scores_dict.keys())
    # 只用完整样本（各模型均不为空）
    valid_mask = np.ones(len(labels), dtype=bool)
    for score in scores_dict.values():
        valid_mask &= ~np.isnan(score)

    scores_clean = {
        k: v[valid_mask] for k, v in scores_dict.items()
    }
    labels_clean = labels[valid_mask]

    df_scores = pd.DataFrame(scores_clean)

    # ── Spearman 秩相关矩阵 ──
    spearman_corr = df_scores.corr(method="spearman")

    # ── 错误相关性矩阵 ──
    errors = {}
    for name, score in scores_clean.items():
        pred = (score > decision_threshold).astype(int)
        errors[name] = (pred != labels_clean).astype(int)
    error_corr = pd.DataFrame(errors).corr()

    # ── 十分位分桶不一致率 ──
    deciles = {}
    for name, score in scores_clean.items():
        deciles[name] = pd.qcut(score, 10, labels=False, duplicates="drop")

    disagree_rate = {}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            key = f"{n1} vs {n2}"
            disagree = (deciles[n1] != deciles[n2]).mean()
            disagree_rate[key] = f"{disagree:.2%}"

    return {
        "spearman_corr":  spearman_corr,
        "error_corr":     error_corr,
        "disagree_rate":  disagree_rate,
        "n_valid_samples": valid_mask.sum(),
    }


def handle_missing_scores(
    scores_matrix: np.ndarray,
    missing_threshold: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    处理模型分中的空值，返回填充后的分数矩阵。

    处理规则：
    - 空值率 < missing_threshold：用列中位数填充，并记录 missing flag
    - 空值率 >= missing_threshold：该模型权重强制为 0（在优化中体现）
    - 全部模型均缺失的样本：标记为需人工审核

    Parameters
    ----------
    scores_matrix : np.ndarray
        原始分数矩阵，shape (N, K)，含 NaN
    missing_threshold : float
        空值率阈值，超过则该模型被排除

    Returns
    -------
    scores_filled : np.ndarray
        填充后的分数矩阵 (N, K)
    missing_flags : np.ndarray
        缺失标志矩阵 (N, K)，1 表示该位置原本为空
    need_manual_review : np.ndarray
        布尔数组 (N,)，True 表示所有模型均无分，需人工审核
    weight_mask : np.ndarray
        权重掩码 (K,)，0 表示该模型空值率过高被排除
    """
    N, K = scores_matrix.shape
    missing_flags  = np.isnan(scores_matrix).astype(float)
    missing_rates  = missing_flags.mean(axis=0)
    scores_filled  = scores_matrix.copy()

    # 权重掩码：空值率过高的模型排除
    weight_mask = (missing_rates < missing_threshold).astype(float)

    for k in range(K):
        col = scores_matrix[:, k]
        col_missing_rate = missing_rates[k]
        median_val = np.nanmedian(col)

        if col_missing_rate >= missing_threshold:
            # 空值率过高：用整列中位数填充（保持数组完整，但权重为0）
            scores_filled[:, k] = np.where(np.isnan(col), median_val, col)
            print(f"  [Missing] 模型{k+1} 空值率={col_missing_rate:.1%} >= {missing_threshold:.0%}，"
                  f"权重强制为0")
        else:
            # 用列中位数填充
            scores_filled[:, k] = np.where(np.isnan(col), median_val, col)

    # 全部模型均缺失的样本
    need_manual_review = (missing_flags.sum(axis=1) == K)
    if need_manual_review.sum() > 0:
        print(f"  [Missing] 全模型缺失样本: {need_manual_review.sum()} 笔，标记为需人工审核")

    return scores_filled, missing_flags, need_manual_review, weight_mask
