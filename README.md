# 信贷风控多模型帕累托策略优化系统

## 功能概述

基于历史多个模型风险分，通过帕累托优化直接计算最优风控拦截策略组合，
无需重训练模型，输出完整的策略指标和跨周期稳定性验证报告。

## 核心指标

| 指标 | 说明 |
|------|------|
| 通过率 | 申请件通过审批的比例 |
| 拒绝率 | 1 - 通过率 |
| 坏账率 | 通过人群中逾期的比例 |
| 风险倍率 | 通过人群坏账率 / 整体申请坏账率，< 1 表示策略有效 |
| 坏账GMV | 通过且逾期样本的贷款金额总和（万元） |
| PSI | 分数分布稳定性，< 0.1 为稳定 |

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

## 输出文件

- `pareto_strategies.csv`：完整 Pareto 策略组合表
- `pareto_dashboard.png`：四图业务决策 Dashboard

## 项目结构

```
credit_strategy_pareto/
├── main.py                    # 主流程
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── data_generator.py     # 多周期数据生成
└── src/
    ├── __init__.py
    ├── metrics.py             # 指标计算
    ├── diagnostics.py         # 空值与多样性诊断
    ├── pareto_optimizer.py    # 帕累托优化核心
    ├── stability_validator.py # 跨周期稳定性验证
    └── strategy_exporter.py  # 策略产出与可视化
```

## 流程说明

```
Step 1  生成历史多周期数据（6个月 × 10000条）
Step 2  空值诊断与处理（MNAR/MAR 识别 + 中位数填充）
Step 3  模型多样性分析（Spearman 相关 + 错误相关 + 不一致率）
Step 4  帕累托优化（ε-约束法，35个前沿点）
Step 5  跨周期稳定性验证（Walk-forward + PSI）
Step 6  策略产出（业务量化 + CSV 导出）
Step 7  可视化 Dashboard（2×2 四图）
Step 8  策略卡片打印
```

## 参数配置

在 `main.py` 顶部的 `CONFIG` 字典中修改所有业务参数：

```python
CONFIG = {
    "target_pass_rate":  0.70,   # 业务目标通过率
    "bad_rate_redline":  0.065,  # 风控红线坏账率
    "current_pass_rate": 0.62,   # 当前策略通过率（基准）
    "current_bad_rate":  0.068,  # 当前策略坏账率（基准）
    "monthly_apply":     50000,  # 月申请量（笔）
    "avg_loan":          20000,  # 平均贷款金额（元）
    "lgd":               0.60,   # 违约损失率
}
```
