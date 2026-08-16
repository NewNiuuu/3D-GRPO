# -*- coding: utf-8 -*-
"""
GRPO 学习信号体检 —— 正式训练前的最后一道关。

GRPO 的梯度完全来自「组内 reward 方差」：grpo_trainer 在 std<=1e-6 时会
`continue` 掉整个 group。所以如果模型在这批题上要么全对要么全错，
frac_nonzero_adv 就会趋近 0，训练能跑但一步都学不到东西。

本脚本用真实数据集里的 prompt 走一遍 采样 -> reward，报告：
  - accuracy              起点准确率
  - frac_nonzero_adv      有多少比例的 group 能产生梯度（最关键）
  - parse_fail_rate       抽不出 A-D 的比例（格式崩了会让 reward 恒 0）
  - tier3_rate            靠最弱的兜底正则 \b([ABCD])\b 命中的比例（可能误判）
  - completion_len        平均生成长度

用法：
    cd /home/aiscuser/nyp/3D-RL
    CUDA_VISIBLE_DEVICES=1 python grpo/probe_signal.py --config grpo/config_test.yaml -n 32
"""
import os
import sys
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
from reward import compute_reward, extract_choice, _PATTERNS


def which_tier(text):
    """返回命中的正则层级（1/2/3），都没命中返回 0。"""
    if not text:
        return 0
    t = text.strip()
    for i, pat in enumerate(_PATTERNS):
        if pat.search(t):
            return i + 1
    return 0


def main():
    ap = argparse.ArgumentParser("GRPO 学习信号体检")
    ap.add_argument("--config", default="grpo/config_test.yaml")
    ap.add_argument("-n", "--num_prompts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=3, help="打印前几条的原始输出")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = load_spatiallm(cfg["model_path"], dtype=getattr(torch, cfg.get("dtype", "bfloat16")))
    model.set_point_backbone_dtype(torch.float32)
    model.config.use_cache = True
    model.to("cuda").eval()

    ds = FloodnetGRPODataset(cfg["train_json"], max_samples=None)
    G = cfg.get("num_generations", 4)
    num_bins = cfg.get("num_bins", 1280)

    # 均匀抽样，避免只覆盖单一子集
    step = max(len(ds) // args.num_prompts, 1)
    idxs = list(range(0, len(ds), step))[: args.num_prompts]

    n_nonzero, n_seen = 0, 0
    acc_sum, len_sum, n_gen = 0.0, 0, 0
    tiers = collections.Counter()
    gt_dist, pred_dist = collections.Counter(), collections.Counter()

    for k, i in enumerate(idxs):
        s = ds[i]
        pcd = load_point_cloud_tensor(s["pcd_path"], num_bins).unsqueeze(0).to("cuda")
        conv = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": s["prompt_text"]},
        ]
        ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        ids = ids.to("cuda")
        plen = ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                input_ids=ids, point_clouds=pcd,
                max_new_tokens=cfg.get("max_new_tokens", 64),
                do_sample=True,
                temperature=cfg.get("temperature", 1.0),
                top_p=cfg.get("top_p", 1.0),
                top_k=cfg.get("top_k", 0) or None,
                num_return_sequences=G, use_cache=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        texts = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
        rewards = torch.tensor([compute_reward(t, s["answer"]) for t in texts])

        n_seen += 1
        if rewards.std() > 1e-6:
            n_nonzero += 1
        acc_sum += rewards.mean().item()
        gt_dist[s["answer"].strip().upper()] += 1
        for t in texts:
            tiers[which_tier(t)] += 1
            pred_dist[extract_choice(t) or "PARSE_FAIL"] += 1
            len_sum += len(tok(t, add_special_tokens=False)["input_ids"])
            n_gen += 1

        if k < args.show:
            print(f"\n--- prompt {i} | GT={s['answer']!r} ---")
            print("  Q:", s["prompt_text"].replace("\n", " ")[:180])
            for j, t in enumerate(texts):
                print(f"  [gen {j}] r={rewards[j].item():.0f} tier={which_tier(t)} | {t.strip()[:100]!r}")

    print("\n" + "=" * 68)
    print(f"prompts={n_seen}  generations={n_gen}  (G={G})")
    print(f"  accuracy            = {acc_sum/n_seen:.3f}   <- 起点准确率")
    print(f"  frac_nonzero_adv    = {n_nonzero/n_seen:.3f}   <- 有梯度的 group 占比【最关键】")
    print(f"  parse_fail_rate     = {tiers[0]/n_gen:.3f}   <- 抽不出 A-D")
    print(f"  tier3_fallback_rate = {tiers[3]/n_gen:.3f}   <- 靠最弱兜底正则命中（可能误判）")
    print(f"  mean_completion_len = {len_sum/n_gen:.1f} tokens")
    print(f"  tier 分布 (1强/2中/3弱/0失败) = {dict(sorted(tiers.items()))}")
    print(f"  GT   分布 = {dict(gt_dist.most_common())}")
    print(f"  预测 分布 = {dict(pred_dist.most_common())}")
    print("=" * 68)
    f = n_nonzero / n_seen
    if f < 0.10:
        print("❌ frac_nonzero_adv < 0.10：绝大多数 group 被跳过，GRPO 基本学不到东西。")
    elif f < 0.30:
        print("⚠️  frac_nonzero_adv 偏低：可考虑调高 temperature 或增大 num_generations。")
    else:
        print("✅ 学习信号充足，可以开训。")


if __name__ == "__main__":
    main()
