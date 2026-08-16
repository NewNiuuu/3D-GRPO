# -*- coding: utf-8 -*-
"""
GRPO 最小试吃 (smoke test) —— 不依赖 trl。

目的：验证 GRPO 的命门能否跑通：
  同一份点云 -> 模型采样出 N 段布局文本 -> 再用 forward 对这 N 段重算 token log-prob。

只要这一步能通，说明"点云透传 + 采样 + 重算概率"这条链路成立，
后续接 trl.GRPOTrainer 只是把这段逻辑接进它的主循环。

用法:
  python grpo/smoke_rollout.py \
    --model_path /home/aiscuser/nyp/ckpts/point_mixed_downsample \
    --point_cloud /Pointcloud-VQA/AirCopBench/Real2_VQA_train/23-00000001_MDMT_when2col_UAV1_1.ply \
    --num_generations 4
"""
import os
import sys
import argparse

import torch
import numpy as np

# 让 `import spatiallm` 可用（脚本在 grpo/ 下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
import spatiallm  # noqa: F401  注册 spatiallm_qwen3 等自定义模型
from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose

# 模型加载逻辑与 trainer 共用同一份（处理 transformers 4.x/5.x 的 dtype 参数
# 差异、以及 meta-device 下 sonata / z-order 的两个坑）。此前这里复制了一份
# 只支持 dtype= 的版本，在 transformers 4.53 上会报
# "__init__() got an unexpected keyword argument 'dtype'"。
from spatiallm_grpo_utils import load_spatiallm

DETECT_TYPE_PROMPT = {
    "all": "Detect walls, doors, windows, boxes.",
    "arch": "Detect walls, doors, windows.",
    "object": "Detect boxes.",
}


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    """复用 inference.py 里完全相同的点云预处理，产出 (1, N, 9) 张量。"""
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    pcd = transform({"name": "pcd", "coord": points.copy(), "color": colors.copy()})
    coord = pcd["grid_coord"]
    xyz = pcd["coord"]
    rgb = pcd["color"]
    feat = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([feat], axis=0))


def build_prompt(tokenizer, model, code_template_file, detect_type="all"):
    with open(code_template_file, "r") as f:
        code_template = f.read()
    task_prompt = DETECT_TYPE_PROMPT[detect_type]
    prompt = (
        f"<|point_start|><|point_pad|><|point_end|>{task_prompt} "
        f"The reference code is as followed: {code_template}"
    )
    if model.config.model_type in ("spatiallm_qwen", "spatiallm_qwen3"):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        conversation = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    # transformers 5.x 下 apply_chat_template 可能返回 BatchEncoding（字典式），
    # 统一取出 input_ids 张量。
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    return input_ids


@torch.no_grad()
def sample_completions(model, tokenizer, input_ids, point_cloud, num_generations,
                       max_new_tokens, temperature, top_p, top_k):
    """对同一份点云采样 num_generations 段文本 (GRPO 的一个 group)。"""
    input_ids = input_ids.to(model.device)
    # num_return_sequences 一次性采样一组；point_clouds 是那根"接线"——手动透传
    out = model.generate(
        input_ids=input_ids,
        point_clouds=point_cloud.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_generations,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return out  # (num_generations, prompt_len + gen_len)


def compute_logprobs(model, tokenizer, full_ids, prompt_len, point_cloud):
    """
    GRPO 重算概率的核心：对采样出的序列，逐条跑一次 forward，
    取每个 completion token 的 log-prob。point_clouds 同样要透传。

    注意：模型 forward 内部会把 <|point_pad|> 展开成很多点云 token，
    序列会变长；这里用最直接的"逐条 forward、对齐 completion 段"的方式验证可行性
    （效率不是这一步的目标，正确性才是）。
    """
    all_logprobs = []
    for i in range(full_ids.shape[0]):
        ids = full_ids[i : i + 1].to(model.device)
        out = model(
            input_ids=ids,
            point_clouds=point_cloud.to(model.device),
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
        logits = out.logits  # (1, L_expanded, V)  L_expanded 因点云插入而变长
        # completion 段的 logits 位于序列末尾，长度 = 生成 token 数
        gen_len = ids.shape[1] - prompt_len
        if gen_len <= 0:
            all_logprobs.append(torch.tensor([], device=model.device))
            continue
        # 预测第 t 个 token 用的是第 t-1 个位置的 logits
        gen_logits = logits[0, -gen_len - 1 : -1, :]
        gen_targets = ids[0, -gen_len:]
        logp = torch.log_softmax(gen_logits.float(), dim=-1)
        tok_logp = logp.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)
        all_logprobs.append(tok_logp.detach().cpu())
    return all_logprobs


def placeholder_reward(text, num_bins):
    """占位 reward：能否被 Layout 解析 + 实体数。够跑通框架用。"""
    try:
        layout = Layout(text)
        layout.undiscretize_and_unnormalize(num_bins=num_bins)
        n = len(layout.get_entities())
        return float(n), n
    except Exception:
        return 0.0, 0


def main():
    ap = argparse.ArgumentParser("SpatialLM GRPO smoke rollout")
    ap.add_argument("--model_path", default="/home/aiscuser/nyp/ckpts/point_mixed_downsample")
    ap.add_argument("--point_cloud", default="/Pointcloud-VQA/AirCopBench/Real2_VQA_train/23-00000001_MDMT_when2col_UAV1_1.ply")
    ap.add_argument("--code_template_file", default="code_template.txt")
    ap.add_argument("--detect_type", default="all", choices=["all", "arch", "object"])
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    print(f"[1/5] 加载模型 {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = load_spatiallm(args.model_path, getattr(torch, args.dtype))
    model.to("cuda")
    model.set_point_backbone_dtype(torch.float32)
    model.eval()
    num_bins = model.config.point_config["num_bins"]

    print(f"[2/5] 读取并预处理点云 {args.point_cloud} ...")
    pcd = load_o3d_pcd(args.point_cloud)
    grid_size = Layout.get_grid_size(num_bins)
    pcd = cleanup_pcd(pcd, voxel_size=grid_size)
    points, colors = get_points_and_colors(pcd)
    point_cloud = preprocess_point_cloud(points, colors, grid_size, num_bins)
    print(f"      点云张量: {tuple(point_cloud.shape)}")

    print("[3/5] 构造 prompt ...")
    input_ids = build_prompt(tokenizer, model, args.code_template_file, args.detect_type)
    prompt_len = input_ids.shape[1]
    print(f"      prompt token 数: {prompt_len}")

    print(f"[4/5] 采样 {args.num_generations} 段 completion ...")
    full_ids = sample_completions(
        model, tokenizer, input_ids, point_cloud,
        args.num_generations, args.max_new_tokens,
        args.temperature, args.top_p, args.top_k,
    )
    texts = tokenizer.batch_decode(
        full_ids[:, prompt_len:], skip_special_tokens=True
    )
    for i, t in enumerate(texts):
        r, n = placeholder_reward(t, num_bins)
        preview = t.replace("\n", " ")[:80]
        print(f"      [gen {i}] reward={r:.1f} entities={n} | {preview} ...")

    print("[5/5] 重算 completion token log-prob（GRPO 命门）...")
    logprobs = compute_logprobs(model, tokenizer, full_ids, prompt_len, point_cloud)
    for i, lp in enumerate(logprobs):
        if lp.numel() == 0:
            print(f"      [gen {i}] 空 completion")
        else:
            print(f"      [gen {i}] {lp.numel()} tokens, mean logp={lp.mean():.4f}")

    print("\n✅ 最小试吃通过：点云透传 + 采样 + 重算 log-prob 全部跑通。")
    print("   下一步即可接 trl.GRPOTrainer，把这段逻辑塞进它的主循环。")


if __name__ == "__main__":
    main()
