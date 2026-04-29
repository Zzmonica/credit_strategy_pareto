"""
metrics.py
信贷风控核心指标计算工具集。
所有函数均为纯函数，无副作用，可独立调用。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def calc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    计算 AUC（Area Under ROC Curve）。

    Parameters
    ----------
    labels : np.ndarray
        真实标签，1=坏账，0=正常
    scores : np.ndarray
        模型预测的风险分（越高风险越高）

    Returns
    -------
    float
        AUC 值，范围 [0.5, 1.0]
    """
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def calc_ks(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    计算 KS 统计量（Kolmogorov-Smirnov）。
    KS = max(TPR - FPR)，衡量模型区分好坏的能力。

    Parameters
    ----------
    labels : np.ndarray
        真实标签
    scores : np.ndarray
        风险分

    Returns
    -------
    float
        KS 值，范围 [0, 1]
    """
    if len(np.unique(labels)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(labels, scores)
    return float((tpr - fpr).max())


def calc_gini(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    计算 Gini 系数。
    Gini = 2 * AUC - 1，与 AUC 等价，范围 [0, 1]。

    Parameters
    ----------
    labels : np.ndarray
        真实标签
    scores : np.ndarray
        风险分

    Returns
    -------
    float
        Gini 系数
    """
    auc = calc_auc(labels, scores)
    return 2 * auc - 1 if not np.isnan(auc) else np.nan


def calc_bad_rate(labels: np.ndarray, pass_mask: np.ndarray) -> float:
    """
    计算通过人群的坏账率。

    Parameters
    ----------
    labels : np.ndarray
        真实标签
    pass_mask : np.ndarray
        布尔数组，True 表示该样本通过审批

    Returns
    -------
    float
        通过人群坏账率
    """
    if pass_mask.sum() == 0:
        return np.nan
    return float(labels[pass_mask].mean())


def calc_rejection_rate(pass_mask: np.ndarray) -> float:
    """
    计算拒绝率（= 1 - 通过率）。

    Parameters
    ----------
    pass_mask : np.ndarray
        布尔数组，True 表示通过

    Returns
    -------
    float
        拒绝率
    """
    return float(1.0 - pass_mask.mean())


def calc_risk_multiplier(
    labels: np.ndarray,
    pass_mask: np.ndarray,
    base_bad_rate: float,
) -> float:
    """
    计算风险倍率。
    风险倍率 = 通过人群坏账率 / 整体申请坏账率。
    倍率 < 1 表示策略有效筛除了高风险人群。

    Parameters
    ----------
    labels : np.ndarray
        真实标签
    pass_mask : np.ndarray
        布尔数组，True 表示通过
    base_bad_rate : float
        整体申请坏账率（基准）

    Returns
    -------
    float
        风险倍率
    """
    br = calc_bad_rate(labels, pass_mask)
    if np.isnan(br) or base_bad_rate == 0:
        return np.nan
    return float(br / base_bad_rate)


def calc_bad_gmv(
    labels: np.ndarray,
    gmv: np.ndarray,
    pass_mask: np.ndarray,
) -> float:
    """
    计算通过人群的坏账 GMV（万元）。
    坏账 GMV = 通过且逾期样本的贷款金额总和。

    Parameters
    ----------
    labels : np.ndarray
        真实标签
    gmv : np.ndarray
        各样本贷款金额（元）
    pass_mask : np.ndarray
        布尔数组，True 表示通过

    Returns
    -------
    float
        坏账 GMV（万元）
    """
    bad_and_pass = pass_mask & (labels == 1)
    return float(gmv[bad_and_pass].sum() / 1e4)


def calc_psi(
    expected_scores: np.ndarray,
    actual_scores: np.ndarray,
    bins: int = 10,
) -> float:
    """
    计算群体稳定性指标 PSI（Population Stability Index）。
    PSI < 0.1   → 稳定
    PSI 0.1~0.2 → 轻微变化，需关注
    PSI >= 0.2  → 显著变化，需重新审视

    Parameters
    ----------
    expected_scores : np.ndarray
        基准期分数（如训练集/验证集）
    actual_scores : np.ndarray
        实际期分数（如线上/OOT）
    bins : int
        分桶数量

    Returns
    -------
    float
        PSI 值
    """
    # 用基准期分位数定义桶边界
    breakpoints = np.nanpercentile(expected_scores, np.linspace(0, 100, bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    def _bucket_pct(scores):
        counts = np.histogram(scores[~np.isnan(scores)], bins=breakpoints)[0]
        pct = counts / counts.sum()
        return np.clip(pct, 1e-6, None)   # 防止除零

    exp_pct = _bucket_pct(expected_scores)
    act_pct = _bucket_pct(actual_scores)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def psi_level(psi: float) -> str:
    """将 PSI 数值转换为稳定性等级描述。"""
    if psi < 0.1:
        return "稳定 ✅"
    elif psi < 0.2:
        return "轻微变化 ⚠️"
    else:
        return "显著变化 🔴"
