# -*- coding: utf-8 -*-
"""
配对显著性检验 —— 判断两个 ckpt 的准确率差距是真的，还是噪声。

为什么需要这个
--------------------------------------------------------------------------
eval_mcq.py 只报总分。n=2000 时 0.9275 vs 0.9230 看着是"掉了 0.45 个点"，
但那只是 9 道题。直接比总分等于假设两次评测相互独立——实际上两个 ckpt 评的是
**同一批题**，绝大多数题两边都答对（或都答错），这些题对差异毫无贡献。

正确做法是只看**分歧的题**（一个对一个错），也就是 McNemar 检验：
    b = A 对 B 错的题数
    c = A 错 B 对的题数
    在"两个模型一样好"的原假设下，每道分歧题归到 b 还是 c 是各 50% 的抛硬币。
若 b+c 很小，哪怕 b-c 看着不小也可能只是抛硬币的正常起伏。

用法
--------------------------------------------------------------------------
    # 先让 eval_mcq 逐条落盘（多个 ckpt 写同一个文件）
    python grpo/eval_mcq.py --config grpo/config_aircop.yaml -n 2000 \
        --dump /tmp/eval.jsonl --ckpt A --ckpt B --ckpt C

    # 再做两两配对检验
    python grpo/compare_ckpts.py /tmp/eval.jsonl
"""
import sys
import json
import math
import argparse
import collections
from itertools import combinations


def mcnemar_exact_p(b, c):
    """双侧精确检验（二项分布 p=0.5）。b+c 大时用正态近似。"""
    n = b + c
    if n == 0:
        return 1.0
    if n <= 1000:
        k = min(b, c)
        # P(X <= k) 的两倍，X ~ Binom(n, 0.5)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
        return min(1.0, 2 * tail)
    z = abs(b - c) / math.sqrt(n)
    return math.erfc(z / math.sqrt(2))


def wilson_ci(k, n, z=1.96):
    """准确率的 Wilson 置信区间，比 p±1.96*se 在接近 0/1 时更靠谱。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    ap = argparse.ArgumentParser("ckpt 配对显著性检验")
    ap.add_argument("dump", help="eval_mcq.py --dump 产出的 jsonl")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    # ckpt -> {idx: reward}
    by_ckpt = collections.defaultdict(dict)
    order = []
    for line in open(args.dump):
        r = json.loads(line)
        if r["ckpt"] not in by_ckpt:
            order.append(r["ckpt"])
        by_ckpt[r["ckpt"]][r["idx"]] = r["reward"]

    if len(order) < 2:
        raise SystemExit(f"只有 {len(order)} 个 ckpt，无法配对比较")

    print("=" * 74)
    print("单个 ckpt 的准确率与 95% 置信区间")
    print("=" * 74)
    for c in order:
        d = by_ckpt[c]
        k, n = sum(d.values()), len(d)
        lo, hi = wilson_ci(k, n)
        print(f"  {c.split('/')[-1]:<34} {k / n:.4f}  [{lo:.4f}, {hi:.4f}]  ({k:.0f}/{n})")
    print("\n  ^ 置信区间互相重叠 = 单看总分区分不出来，必须看下面的配对检验")

    print("\n" + "=" * 74)
    print("两两配对检验（McNemar，只统计分歧的题）")
    print("=" * 74)
    for a, b in combinations(order, 2):
        da, db = by_ckpt[a], by_ckpt[b]
        common = sorted(set(da) & set(db))
        n_b = sum(1 for i in common if da[i] > db[i])  # a 对 b 错
        n_c = sum(1 for i in common if da[i] < db[i])  # a 错 b 对
        p = mcnemar_exact_p(n_b, n_c)
        both = len(common) - n_b - n_c
        verdict = "✅ 差异显著" if p < args.alpha else "❌ 无显著差异（就是噪声）"
        print(f"\n  {a.split('/')[-1]}  vs  {b.split('/')[-1]}")
        print(f"    共同题 {len(common)}，其中两边一致 {both} 题（{both / len(common):.1%}），分歧 {n_b + n_c} 题")
        print(f"    前者独对 b={n_b}   后者独对 c={n_c}   净差 {n_b - n_c:+d}")
        print(f"    p = {p:.4f}   {verdict}")
        if p >= args.alpha and n_b + n_c > 0:
            # 要达到显著大概需要多少分歧样本
            need = math.ceil((1.96 * math.sqrt(n_b + n_c)) ** 2 / max(abs(n_b - n_c), 1) ** 2 * (n_b + n_c))
            print(f"    （按当前效应量，约需 {need} 道分歧题才可能显著；加大 -n 或改进方法）")

    print("\n" + "=" * 74)
    print("怎么读：b+c 就是全部有效信息量。两边一致的题再多也不提供任何区分度。")
    print("=" * 74)


if __name__ == "__main__":
    main()
