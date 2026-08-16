# -*- coding: utf-8 -*-
"""
选择题 reward（单选）。

数据形如：
    question: "... A. xxx B. yyy C. zzz D. www"
    answer:   "C"          # GT 单字母
    模型输出:  "C" / "The answer is B." / "b" ...

reward 规则（0/1，纯精确匹配）：
  - 从模型输出里抽出它选的字母，与 GT 严格相等 → 1.0
  - 抽不到字母（未按格式作答）或选错 → 0.0

抽取失败（parse fail）与"选错"都记 0，但通过 extract_choice 返回 None 可区分：
上层若想统计"格式失败率"，可自行调用 extract_choice 判 None。

接口保持 (completion_text, answer) -> float，与旧版一致，无需改 trainer 调用。

--------------------------------------------------------------------------
选项范围（2026-08-16 扩展）
--------------------------------------------------------------------------
原版正则写死 [ABCD]。AirCopBench 确实只有 A-D，但 UrbanVideoBench 的题目最多
到 G（实测 GT 分布 A:739 B:890 C:781 D:632 E:806 F:53 G:179），其中 E/F/G 共
1038 条 = 25.4%。这些题在原版下 normalize_gt 返回 None → reward 恒 0 →
组内无方差 → 被 grpo_trainer 的 std<=1e-6 分支整组跳过，等于 1/4 数据白跑。

因此把范围放宽到 A-H（题面实测最大到 H）。0/1 的判定语义完全没变。

副作用：兜底正则的误命中面变大了（多了 E-H 四个字母）。当前 ckpt 输出就是
单个字母，实测 parse_fail_rate=0，不受影响；若将来模型开始输出整句，用
grpo/probe_signal.py 复测 tier3_fallback_rate。
"""
import re

# 有效选项字母范围。数据里最大到 H；改这里即可整体调整。
CHOICES = "ABCDEFGH"
_C = f"[{CHOICES}]"

# 匹配优先级：从强到弱，命中即返回。
#   1) 明确措辞："answer is B" / "答案是 B" / "option B" / "choice: B"
#   2) 行首/串首单独一个字母，可带 . ) 、顿号 等："B."  "B)"  "(B)"  "B、"
#   3) 兜底：全文出现的第一个孤立字母（\b 词边界，避免命中单词里的字母）
_PATTERNS = [
    re.compile(
        r"(?:answer|choice|option|答案|选项|选择)\s*(?:is|:|：|为|是)?\s*[\(\[]?\s*("
        + _C
        + r")\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*[\(\[]?\s*(" + _C + r")\s*[\)\]\.\、\:：]", re.IGNORECASE),
    re.compile(r"\b(" + _C + r")\b", re.IGNORECASE),
]


def extract_choice(text: str):
    """从模型输出里抽取选项字母，返回大写字母，抽不到返回 None。"""
    if not text:
        return None
    t = text.strip()
    for pat in _PATTERNS:
        m = pat.search(t)
        if m:
            return m.group(1).upper()
    return None


def normalize_gt(answer: str):
    """把 GT 归一成大写单字母；不在 CHOICES 范围内返回 None。"""
    if not answer:
        return None
    m = re.search(_C, answer.strip(), re.IGNORECASE)
    return m.group(0).upper() if m else None


def compute_reward(completion_text: str, answer: str) -> float:
    gt = normalize_gt(answer)
    pred = extract_choice(completion_text)
    if gt is None or pred is None:
        return 0.0
    return 1.0 if pred == gt else 0.0
