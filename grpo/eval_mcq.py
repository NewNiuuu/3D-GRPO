# -*- coding: utf-8 -*-
"""
选择题 checkpoint 评测 —— 训完之后拿它看有没有提升。

为什么不用仓库根目录的 eval.py：那是 SpatialLM 原版的**室内布局估计**评测
（wall/door/window + 20 类家具的 F1@IoU），输入是 layout txt 文件对，
和选择题准确率毫无关系。inference.py 同理（generate_layout）。

与 probe_signal.py 的分工：
  probe_signal  开训**前**用，随机采样 G 条，看 frac_nonzero_adv 够不够学
  eval_mcq      训完**后**用，贪心解码单条，看 accuracy 涨没涨

关键差别是 **do_sample=False**：probe_signal 走随机采样，同一个 ckpt 每次跑
结果都不一样，不能用来做 ckpt 之间的比较。这里必须贪心，结果可复现。

⚠️ 没有留出集
--------------------------------------------------------------------------
data/ 下两个数据集都只有 train split（AirCop 四个 *_VQA_train.json、
UrbanVideo 一个 MCQ json），且训练配置是 max_samples: null，即全量参与训练。
所以本脚本抽的子集**是训练集的子集**，测出来的 accuracy 涨了只能说明
"模型没被训坏 / 在见过的题上更准了"，**不能证明泛化**。
真要证明泛化，得先切一份不参与训练的留出集重训。

用法
--------------------------------------------------------------------------
    cd /home/aiscuser/nyp/3D-RL

    # 单个 ckpt
    CUDA_VISIBLE_DEVICES=0 python grpo/eval_mcq.py \
        --config grpo/config_aircop.yaml -n 512

    # 多个 ckpt 对比（跑在完全相同的子集上，最后出对比表）
    CUDA_VISIBLE_DEVICES=0 python grpo/eval_mcq.py \
        --config grpo/config_aircop.yaml -n 512 \
        --ckpt /home/aiscuser/nyp/ckpts/point_mixed_downsample \
        --ckpt /home/aiscuser/nyp/saves/grpo_aircop/checkpoint-200 \
        --ckpt /home/aiscuser/nyp/saves/grpo_aircop
"""
import os
import sys
import time
import json
import random
import argparse
import collections

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spatiallm  # noqa: F401  注册自定义模型
from transformers import AutoTokenizer
from spatiallm_grpo_utils import load_spatiallm, load_point_cloud_tensor
from dataset import FloodnetGRPODataset
from reward import compute_reward, extract_choice


def split_of(pcd_path: str) -> str:
    """从点云路径推出它属于哪个子集，用于分项拆解。

    AirCop    .../AirCopBench/Real2_VQA_train/xxx.ply   -> 父目录有区分度
    UrbanVideo .../train_64/EmbodiedCity_1.ply          -> 父目录只有一个，
                                                           改用文件名前缀
    调用方按整个数据集的父目录种类数决定用哪种，见 pick_split_fn。
    """
    return os.path.basename(os.path.dirname(pcd_path))


def prefix_of(pcd_path: str) -> str:
    return os.path.basename(pcd_path).split("_")[0]


def pick_split_fn(samples):
    """父目录能区分就用父目录，否则退回文件名前缀。"""
    if len({split_of(s["pcd_path"]) for s in samples}) > 1:
        return split_of
    return prefix_of


def pct(n, d):
    return f"{n / d:.4f}" if d else "  n/a "


@torch.no_grad()
def run_one(ckpt, cfg, subset, tok, device="cuda", sample_seed=0):
    """在给定子集上评一个 ckpt，返回逐条结果 list[dict]。"""
    model = load_spatiallm(ckpt, dtype=getattr(torch, cfg.get("dtype", "bfloat16")))
    model.set_point_backbone_dtype(torch.float32)
    model.config.use_cache = True
    model.to(device).eval()

    num_bins = cfg.get("num_bins", 1280)
    max_points = cfg.get("max_points", 0)
    max_new = cfg.get("max_new_tokens", 8)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # 点云 LRU：AirCop 935 朵云 / 13578 条样本，抽样时会有重复
    cache = collections.OrderedDict()
    cache_cap = int(cfg.get("pcd_cache_size", 32))

    def get_pcd(path):
        if path in cache:
            cache.move_to_end(path)
            return cache[path]
        # sample_seed 必传：max_points 触发封顶时（UrbanVideo 实测 62% 的点云会触发），
        # 抽哪 16384 个点必须只由 (seed, path) 决定。否则各 ckpt 在各自的进程里
        # 用全局 torch RNG 各抽各的，base 和 ckpt 面对的是**不同的点云**，
        # 对比结果里就混进了重采样噪声，说不清预测差异是不是模型带来的。
        t = load_point_cloud_tensor(
            path, num_bins, max_points=max_points, sample_seed=sample_seed
        )
        cache[path] = t
        while len(cache) > cache_cap:
            cache.popitem(last=False)
        return t

    rows = []
    t0 = time.time()
    for k, s in enumerate(subset):
        pcd = get_pcd(s["pcd_path"]).unsqueeze(0).to(device)
        conv = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": s["prompt_text"]},
        ]
        ids = tok.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        ids = ids.to(device)
        plen = ids.shape[1]

        out = model.generate(
            input_ids=ids,
            point_clouds=pcd,
            max_new_tokens=max_new,
            do_sample=False,  # 贪心，保证可复现、可比较
            use_cache=True,
            pad_token_id=pad_id,
        )
        text = tok.batch_decode(out[:, plen:], skip_special_tokens=True)[0]

        rows.append(
            {
                "split": s["_split"],
                "gt": s["answer"].strip().upper(),
                "pred": extract_choice(text),
                "reward": compute_reward(text, s["answer"]),
                "n_tok": len(tok(text, add_special_tokens=False)["input_ids"]),
                "text": text,
            }
        )
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(
                f"    {k + 1}/{len(subset)}  acc={sum(r['reward'] for r in rows) / len(rows):.4f}"
                f"  {el / (k + 1):.2f}s/条  ETA {(len(subset) - k - 1) * el / (k + 1) / 60:.1f}min",
                flush=True,
            )

    del model
    torch.cuda.empty_cache()
    return rows


def report(ckpt, rows, show=0):
    n = len(rows)
    acc = sum(r["reward"] for r in rows) / n
    fail = sum(1 for r in rows if r["pred"] is None) / n
    mlen = sum(r["n_tok"] for r in rows) / n

    print("\n" + "=" * 72)
    print(f"ckpt: {ckpt}")
    print("=" * 72)
    print(f"  overall accuracy     = {acc:.4f}  ({sum(r['reward'] for r in rows):.0f}/{n})")
    print(f"  parse_fail_rate      = {fail:.4f}   <- 抽不出选项字母的比例")
    print(f"  mean_completion_len  = {mlen:.1f} tokens")

    # 众数基线：无脑全答最常见的那个字母能拿多少分。低于它 = 还不如瞎猜
    gt_cnt = collections.Counter(r["gt"] for r in rows)
    maj_letter, maj_n = gt_cnt.most_common(1)[0]
    print(f"  众数基线（全答 {maj_letter}）  = {maj_n / n:.4f}   <- 低于这个说明模型没在做题")

    # 分项 ①：按数据集 split
    by_split = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        by_split[r["split"]][0] += r["reward"]
        by_split[r["split"]][1] += 1
    print("\n  按数据集 split:")
    for k, (s, c) in sorted(by_split.items(), key=lambda kv: -kv[1][1]):
        print(f"    {k:<20} {pct(s, c)}   ({s:.0f}/{c})")

    # 分项 ②：按 GT 字母。能看出是不是塌向某个选项
    by_gt = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        by_gt[r["gt"]][0] += r["reward"]
        by_gt[r["gt"]][1] += 1
    print("\n  按 GT 字母（该字母的题答对了多少）:")
    for k in sorted(by_gt):
        s, c = by_gt[k]
        print(f"    {k:<20} {pct(s, c)}   ({s:.0f}/{c})")

    pred_cnt = collections.Counter(r["pred"] or "PARSE_FAIL" for r in rows)
    print(f"\n  GT   分布 = {dict(sorted(gt_cnt.items()))}")
    print(f"  预测 分布 = {dict(sorted(pred_cnt.items(), key=lambda kv: str(kv[0])))}")
    print("  ^ 两行差距越大越说明模型有选项偏好（塌向某个字母是训崩的典型征兆）")

    for r in rows[:show]:
        print(f"\n  [样例] GT={r['gt']} pred={r['pred']} r={r['reward']:.0f} | {r['text'].strip()[:100]!r}")

    return {"acc": acc, "fail": fail, "len": mlen, "by_split": dict(by_split)}


def main():
    ap = argparse.ArgumentParser("选择题 ckpt 评测")
    ap.add_argument("--config", required=True, help="复用训练 config 的数据/解码设置")
    ap.add_argument(
        "--ckpt",
        action="append",
        default=[],
        help="要评的 ckpt，可重复传多个做对比；不传则用 config 里的 model_path",
    )
    ap.add_argument("-n", "--num_prompts", type=int, default=512, help="抽多少条；-1=全量")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="同时决定两件事：①抽哪些条 ②max_points 封顶时每个点云抽哪些点。"
        "两者都必须在各 ckpt 之间保持一致，对比才有意义。",
    )
    ap.add_argument("--show", type=int, default=0, help="打印前几条原始输出")
    ap.add_argument(
        "--dump",
        default=None,
        help="把逐条结果写到这个 jsonl。多 ckpt 时才能做配对显著性检验——"
        "总分差几个百分点常常只是噪声，必须看同一道题上谁对谁错（见 compare_ckpts.py）",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ckpts = args.ckpt or [cfg["model_path"]]
    for c in ckpts:
        if not os.path.isdir(c):
            raise SystemExit(f"ckpt 不存在: {c}")

    tok = AutoTokenizer.from_pretrained(cfg["model_path"])
    ds = FloodnetGRPODataset(cfg["train_json"], max_samples=None)

    # 固定 seed 抽子集：所有 ckpt 必须评在**完全相同**的题上，否则对比无意义
    idxs = list(range(len(ds)))
    if args.num_prompts > 0 and args.num_prompts < len(ds):
        idxs = sorted(random.Random(args.seed).sample(idxs, args.num_prompts))
    subset = [ds[i] for i in idxs]
    split_fn = pick_split_fn(subset)
    for s in subset:
        s["_split"] = split_fn(s["pcd_path"])

    print(f"数据集 {cfg['train_json']}：全量 {len(ds)} 条，本次评测 {len(subset)} 条 (seed={args.seed})")
    print(f"解码：贪心 do_sample=False, max_new_tokens={cfg.get('max_new_tokens', 8)}")
    print("⚠ 抽自训练集，accuracy 上升不能证明泛化——详见本文件顶部说明。\n")

    summaries = []
    for c in ckpts:
        print(f"\n>>> 评测 {c} ...", flush=True)
        rows = run_one(c, cfg, subset, tok, sample_seed=args.seed)
        summaries.append((c, report(c, rows, show=args.show)))
        if args.dump:
            # 逐条落盘：idx 是数据集里的全局下标，配对检验靠它对齐
            with open(args.dump, "a") as f:
                for i, r in zip(idxs, rows):
                    f.write(
                        json.dumps(
                            {
                                "ckpt": c,
                                "idx": i,
                                "split": r["split"],
                                "gt": r["gt"],
                                "pred": r["pred"],
                                "reward": r["reward"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(f"  [dump] 逐条结果已追加到 {args.dump}")

    if len(summaries) > 1:
        print("\n" + "=" * 72)
        print("对比（同一批题、同一解码设置）")
        print("=" * 72)
        base = summaries[0][1]["acc"]
        print(f"  {'ckpt':<52} {'acc':>8} {'Δ vs 第一个':>12}")
        for c, s in summaries:
            d = s["acc"] - base
            print(f"  {os.path.basename(c.rstrip('/')):<52} {s['acc']:>8.4f} {d:>+12.4f}")
        print("\n  提醒：n 越小噪声越大。n=512 时 1 个百分点约等于 5 条题的抖动，")
        print("  想区分 1~2 个点的差距建议 -n 2000 以上。")


if __name__ == "__main__":
    main()
