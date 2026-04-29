"""
data_generator.py
生成模拟的多周期历史信贷申请数据，用于帕累托策略优化系统的测试与演示。
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_monthly_data(
    n_months: int = 6,
    n_samples_per_month: int = 10000,
    base_bad_rate: float = 0.12,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    生成多周期历史信贷数据。

    Parameters
    ----------
    n_months : int
        生成月份数，默认 6 个月
    n_samples_per_month : int
        每月样本量
    base_bad_rate : float
        基础坏账率（申请件口）
    random_seed : int
        随机种子，保证可复现

    Returns
    -------
    pd.DataFrame
        包含多周期历史数据的 DataFrame
    """
    np.random.seed(random_seed)

    # 起始月份
    start_month = pd.Period("2025-07", freq="M")
    months = [start_month + i for i in range(n_months)]

    all_data = []

    for month_idx, month in enumerate(months):
        # ── 1. 坏账率随时间轻微漂移（模拟真实场景） ──
        drift = month_idx * 0.003          # 每月增加 0.3%
        month_bad_rate = base_bad_rate + drift

        # ── 2. 生成基础特征与标签 ──
        X, y = make_classification(
            n_samples=n_samples_per_month,
            n_features=15,
            n_informative=8,
            n_redundant=3,
            weights=[1 - month_bad_rate, month_bad_rate],
            flip_y=0.01,            # 轻微标签噪声
            random_state=random_seed + month_idx,
        )

        # ── 3. 生成三版本模型分（风险概率，越高风险越高） ──
        # 基础分：用不同特征子集的线性组合模拟不同模型
        base_score = _sigmoid(X[:, :8] @ np.random.randn(8) * 0.5)

        score_v1 = _add_model_noise(base_score, y, noise_std=0.08, seed=random_seed + 0)
        score_v2 = _add_model_noise(base_score, y, noise_std=0.10, seed=random_seed + 1)
        score_v3 = _add_model_noise(base_score, y, noise_std=0.09, seed=random_seed + 2)

        # ── 4. 注入随机空值（各模型空值位置不完全相同） ──
        score_v1 = _inject_missing(score_v1, missing_rate=0.07, seed=random_seed + 10 + month_idx)
        score_v2 = _inject_missing(score_v2, missing_rate=0.12, seed=random_seed + 20 + month_idx)
        score_v3 = _inject_missing(score_v3, missing_rate=0.05, seed=random_seed + 30 + month_idx)

        # ── 5. 生成 GMV（对数正态分布，5000~100000） ──
        gmv = np.random.lognormal(mean=10.0, sigma=0.8, size=n_samples_per_month)
        gmv = np.clip(gmv, 5000, 100000).round(-2)   # 取整到百元

        # ── 6. 用户分群 ──
        segment_probs = [0.35, 0.50, 0.15]           # new / existing / high_value
        segment = np.random.choice(
            ["new", "existing", "high_value"],
            size=n_samples_per_month,
            p=segment_probs,
        )

        # ── 7. 组装 DataFrame ──
        df_month = pd.DataFrame({
            "apply_month":  str(month),
            "user_id":      [f"{month}_{i:05d}" for i in range(n_samples_per_month)],
            "label":        y,
            "gmv":          gmv,
            "score_v1":     score_v1,
            "score_v2":     score_v2,
            "score_v3":     score_v3,
            "segment":      segment,
        })

        all_data.append(df_month)
        print(f"  [DataGen] {month} | 样本量={n_samples_per_month} | "
              f"坏账率={y.mean():.2%} | "
              f"V1空值率={np.isnan(score_v1).mean():.1%} | "
              f"V2空值率={np.isnan(score_v2).mean():.1%} | "
              f"V3空值率={np.isnan(score_v3).mean():.1%}")

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"\n  [DataGen] 总数据量: {len(full_df):,} 条 | "
          f"整体坏账率: {full_df['label'].mean():.2%}\n")
    return full_df


# ══════════════════════════════════════════
# 内部辅助函数
# ══════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数，将任意实数映射到 (0, 1)。"""
    return 1.0 / (1.0 + np.exp(-x))


def _add_model_noise(
    base_score: np.ndarray,
    labels: np.ndarray,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """
    在基础分上叠加噪声，模拟不同模型版本的预测差异。
    坏账样本的分数整体偏高，好样本偏低，但各版本偏移量略有不同。
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(base_score)) * noise_std
    # 坏账样本额外正偏移，强化区分度
    bias = labels * rng.uniform(0.05, 0.15)
    raw = base_score + noise + bias
    return np.clip(_sigmoid(raw * 2 - 1), 0.001, 0.999)


def _inject_missing(
    scores: np.ndarray,
    missing_rate: float,
    seed: int,
) -> np.ndarray:
    """随机将部分分数置为 NaN，模拟数据缺失。"""
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(scores)) < missing_rate
    result = scores.copy().astype(float)
    result[mask] = np.nan
    return result
