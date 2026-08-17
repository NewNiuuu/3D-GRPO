# -*- coding: utf-8 -*-
"""
聚合训练时打的难度标记 -> 数据集难度图谱 + 难题子集。

数据来自 config 里 difficulty_log: true 时写出的
    <output_dir>/difficulty/rank{0..7}.jsonl
每行是一个 GRPO group 的逐题结果：idx / step / epoch / n_correct / n_gen / preds。
跑 5 个 epoch 就是每题被采样 5 次 × 8 条 = 40 条 rollout，比单独跑一遍
probe_signal.py 的统计量足得多，而且完全免费——训练本来就在算这些。

用法：
    # 看全数据集难度分布
    python grpo/analyze_difficulty.py --dir /home/aiscuser/nyp/saves/grpo_urbanvideo_holdout/difficulty

    # 只用第一个 epoch 的数据（此时策略最接近 base，难度估计不受训练污染）
    python grpo/analyze_difficulty.py --dir ... --max_epoch 1.0

    # 导出难题子集，供下一轮训练
    python grpo/analyze_difficulty.py --dir ... --max_epoch 1.0 \
        --src data/UrbanVideoBench/MCQ_EmbodiedCity_AerialVLN_train.json \
        --emit_hard data/UrbanVideoBench/hard_train.json

⚠ 一个必须注意的偏差：p_correct 是**跨 epoch 混算**的，而策略在训练中一直在变。
  若模型真的学到了东西，后面 epoch 的正确率天然更高，混在一起算会把题目显得偏简单。
  所以做难题筛选时建议加 --max_epoch 1.0，只用第一遍的数据。
"""
import argparse
import collections
import glob
import json
import os


def load(dir_path, max_epoch=None):
    rows = []
    files = sorted(glob.glob(os.path.join(dir_path, "rank*.jsonl")))
    if not files:
        raise SystemExit(f"{dir_path} 下没有 rank*.jsonl，确认 config 里 difficulty_log: true")
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 训练还在跑时最后一行可能只写了一半
                if max_epoch is not None and r.get("epoch", 0) > max_epoch:
                    continue
                rows.append(r)
    print(f"读入 {len(rows)} 条 group 记录（来自 {len(files)} 个 rank 文件）"
          + (f"，只取 epoch <= {max_epoch}" if max_epoch is not None else ""))
    return rows


def main():
    ap = argparse.ArgumentParser("聚合难度标记")
    ap.add_argument("--dir", required=True, help="<output_dir>/difficulty 目录")
    ap.add_argument("--max_epoch", type=float, default=None,
                    help="只统计 epoch <= 该值的记录。做难题筛选时建议设 1.0")
    ap.add_argument("--src", default=None, help="原始训练 json，用于导出子集")
    ap.add_argument("--emit_hard", default=None, help="把难题子集写到这个 json")
    ap.add_argument("--keep_allwrong", action="store_true",
                    help="导出时把「全错」的题也留下（默认剔除，见下方说明）")
    ap.add_argument("--show", type=int, default=0, help="打印几道最难的题看看")
    args = ap.parse_args()

    rows = load(args.dir, args.max_epoch)

    # 按题聚合
    agg = collections.defaultdict(lambda: {"ok": 0, "n": 0, "preds": [], "gt": None, "pcd": None})
    for r in rows:
        a = agg[r["idx"]]
        a["ok"] += r["n_correct"]
        a["n"] += r["n_gen"]
        a["gt"] = r.get("gt")
        a["pcd"] = r.get("pcd")
        a["preds"].extend(r.get("preds", []))

    print(f"覆盖 {len(agg)} 道题，累计 {sum(a['n'] for a in agg.values())} 条 rollout")

    # 分桶。p = 该题累计答对数 / 累计 rollout 数。
    # 只有中间两桶（0 < p < 1）组内 reward 有方差、advantage 非零，才真正产生梯度；
    # 两头是 std=0 被 compute_loss 直接 continue 掉的，训了等于没训。
    B_EASY0 = "太易·零梯度 (p=1)"
    B_EASY = "偏易 (0.5<=p<1)"
    B_HARD = "偏难 (0<p<0.5)"
    B_HARD0 = "太难·零梯度 (p=0)"
    buckets = {B_EASY0: [], B_EASY: [], B_HARD: [], B_HARD0: []}
    for idx, a in agg.items():
        p = a["ok"] / a["n"]
        if p >= 1.0:
            buckets[B_EASY0].append(idx)
        elif p <= 0.0:
            buckets[B_HARD0].append(idx)
        elif p >= 0.5:
            buckets[B_EASY].append(idx)
        else:
            buckets[B_HARD].append(idx)

    tot = len(agg)
    print("\n难度分布：")
    for k, v in buckets.items():
        print(f"  {k:<22} {len(v):>6} 题  {len(v)/tot*100:>5.1f}%")
    dead = len(buckets[B_EASY0]) + len(buckets[B_HARD0])
    print(f"\n  → 其中 {dead} 题（{dead/tot*100:.1f}%）组内零方差，**全程不产生任何梯度**，"
          f"这部分算力完全是白烧的。")

    # 全错的题要单独看：可能是真难，也可能是标注错/答案格式对不上。
    # 判据：错的时候是不是总错成同一个选项。
    if buckets[B_HARD0]:
        systematic = 0
        for idx in buckets[B_HARD0]:
            a = agg[idx]
            c = collections.Counter(p for p in a["preds"] if p)
            if c and c.most_common(1)[0][1] / max(1, len(a["preds"])) >= 0.8:
                systematic += 1
        n_aw = len(buckets[B_HARD0])
        print(f"\n  全错的 {n_aw} 题里，{systematic} 题（{systematic/n_aw*100:.0f}%）"
              f"是**稳定错成同一个选项**。")
        print("    这类不是「不会」，而是模型有确定的错误信念，或该题 GT 本身可疑；")
        print("    RL 靠采样探索很难纠正（8 条 rollout 一条都没采到对的，advantage 恒为 0）。")
        print("    要么先人工抽查 GT，要么这批题得靠 SFT 而不是 RL。")

    # 按来源拆
    print("\n按来源：")
    per_src = collections.defaultdict(lambda: [0, 0])
    for idx, a in agg.items():
        tag = os.path.basename(str(a["pcd"] or "")).split("_")[0] or "unknown"
        per_src[tag][0] += a["ok"]
        per_src[tag][1] += a["n"]
    for tag, (ok, n) in sorted(per_src.items()):
        print(f"  {tag:<14} 平均正确率 {ok/max(n,1):.4f}  ({n} 条 rollout)")

    if args.show:
        print(f"\n最难的 {args.show} 道（按正确率升序）：")
        order = sorted(agg.items(), key=lambda kv: (kv[1]["ok"] / kv[1]["n"], kv[0]))
        for idx, a in order[: args.show]:
            c = collections.Counter(p for p in a["preds"] if p)
            print(f"  idx={idx:<6} p={a['ok']/a['n']:.3f}  GT={a['gt']}  "
                  f"预测分布={dict(c.most_common(4))}")

    # 导出难题子集
    if args.emit_hard:
        if not args.src:
            raise SystemExit("--emit_hard 需要同时给 --src 指定原始 json")
        keep = set(buckets[B_HARD]) | set(buckets[B_EASY])
        if args.keep_allwrong:
            keep |= set(buckets[B_HARD0])
        data = json.load(open(args.src))
        sub = [data[i] for i in sorted(keep) if 0 <= i < len(data)]
        with open(args.emit_hard, "w") as f:
            json.dump(sub, f, ensure_ascii=False)
        print(f"\n[out] {args.emit_hard}: {len(sub)} 条（原 {len(data)} 条，留下 {len(sub)/len(data)*100:.1f}%）")
        print("  剔除了「全对」的题——它们组内零方差，训了也不产生梯度，只是在烧采样算力。")
        if not args.keep_allwrong:
            print("  也剔除了「全错」的题（同样零梯度）。若想保留，加 --keep_allwrong，")
            print("  但先确认它们不是 GT 有问题——见上面的「稳定错成同一选项」比例。")
        print("\n⚠ 别直接用它替换全量训练集：把简单题全删掉，模型可能在这些题上退化"
              "（灾难性遗忘）。稳妥做法是难题过采样而不是删——例如难题重复 2~3 份"
              "再和原始数据拼起来，让有梯度的组占比上去，同时保留简单题做锚。")


if __name__ == "__main__":
    main()
