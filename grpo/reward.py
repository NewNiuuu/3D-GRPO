# -*- coding: utf-8 -*-
"""
占位 reward（先跑通框架用，不追求 reward 质量）。

组合两部分：
  - 格式分：答案非空、长度合理（不复读、不空）
  - 命中分：预测里是否包含 GT 答案（大小写不敏感）

后续要做真 reward（如 VQA 准确率、布局 F1）只需替换 compute_reward 即可，
接口保持 (completion_text, answer) -> float。
"""
import re


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def compute_reward(completion_text: str, answer: str) -> float:
    pred = _normalize(completion_text)
    gt = _normalize(answer)

    reward = 0.0
    # 格式分：非空且不过长（避免退化成复读）
    if 0 < len(pred) <= 200:
        reward += 0.2
    # 命中分：预测包含 GT（简单子串匹配，够占位）
    if gt and gt in pred:
        reward += 1.0
    return reward
