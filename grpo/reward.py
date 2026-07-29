# -*- coding: utf-8 -*-
"""
选择题 reward（A/B/C/D 单选）。

数据形如：
    question: "... A. xxx B. yyy C. zzz D. www"
    answer:   "C"          # GT 单字母
    模型输出:  "C" / "The answer is B." / "b" ...

reward 规则（0/1，纯精确匹配，先跑通流程最合适）：
  - 从模型输出里抽出它选的字母（A-D），与 GT 严格相等 → 1.0
  - 抽不到字母（未按格式作答）或选错 → 0.0

抽取失败（parse fail）与"选错"都记 0，但通过 extract_choice 返回 None 可区分：
上层若想统计"格式失败率"，可自行调用 extract_choice 判 None。

接口保持 (completion_text, answer) -> float，与旧版一致，无需改 trainer 调用。
"""
import re

# 匹配优先级：从强到弱，命中即返回。
#   1) 明确措辞："answer is B" / "答案是 B" / "option B" / "choice: B"
#   2) 行首/串首单独一个字母，可带 . ) 、顿号 等："B."  "B)"  "(B)"  "B、"
#   3) 兜底：全文出现的第一个孤立 A-D 字母（\b 词边界，避免命中单词里的字母）
_PATTERNS = [
    re.compile(
        r"(?:answer|choice|option|答案|选项|选择)\s*(?:is|:|：|为|是)?\s*[\(\[]?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*[\(\[]?\s*([ABCD])\s*[\)\]\.\、\:：]", re.IGNORECASE),
    re.compile(r"\b([ABCD])\b", re.IGNORECASE),
]


def extract_choice(text: str):
    """从模型输出里抽取选项字母，返回大写 'A'/'B'/'C'/'D'，抽不到返回 None。"""
    if not text:
        return None
    t = text.strip()
    for pat in _PATTERNS:
        m = pat.search(t)
        if m:
            return m.group(1).upper()
    return None


def normalize_gt(answer: str):
    """把 GT 归一成大写单字母；非 A-D 返回 None。"""
    if not answer:
        return None
    m = re.search(r"([ABCD])", answer.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None


def compute_reward(completion_text: str, answer: str) -> float:
    gt = normalize_gt(answer)
    pred = extract_choice(completion_text)
    if gt is None or pred is None:
        return 0.0
    return 1.0 if pred == gt else 0.0
