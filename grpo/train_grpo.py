# -*- coding: utf-8 -*-
"""
SpatialLM GRPO 训练入口（HF Trainer + 自定义 GRPO loss）。

四卡启动示例（在 uavlm 环境）：
    cd /root/lnj/SpatialLM
    torchrun --nproc_per_node 4 grpo/train_grpo.py --config grpo/config_floodnet.yaml

单卡快速验证：
    CUDA_VISIBLE_DEVICES=0 python grpo/train_grpo.py --config grpo/config_floodnet.yaml --max_steps 3
"""
import os
import sys
import argparse
import logging as _logging

import torch
import yaml
from transformers import AutoTokenizer, TrainingArguments

# 训练时开了 gradient_checkpointing，transformers 会在每层解码器打印
# "Caching is incompatible with gradient checkpointing ... past_key_value=None"。
# 这只是提示(训练本就不需要 KV cache)，无害但刷屏。这里把它静音。
_logging.getLogger("transformers.models.qwen3.modeling_qwen3").setLevel(_logging.ERROR)
_logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)
import warnings as _warnings

_warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spatiallm  # noqa: F401  注册自定义模型
from spatiallm_grpo_utils import load_spatiallm
from dataset import FloodnetGRPODataset
from grpo_trainer import SpatialLMGRPOTrainer


def main():
    ap = argparse.ArgumentParser("SpatialLM GRPO trainer")
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument("--max_steps", type=int, default=None, help="覆盖 config 的 max_steps（调试用）")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

    # ---- 模型 + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = load_spatiallm(cfg["model_path"], dtype=dtype)
    model.set_point_backbone_dtype(torch.float32)  # 点云编码器保持 fp32（与推理一致）
    model.config.use_cache = False  # 训练关掉 KV cache

    # ---- reference model（冻结，用于 KL 惩罚）----
    # 默认从同一 SFT ckpt 再加载一份作为 π_ref。kl_coef=0 时可关闭以省显存。
    ref_model = None
    if float(cfg.get("kl_coef", 0.04)) > 0:
        ref_model = load_spatiallm(cfg["model_path"], dtype=dtype)
        ref_model.set_point_backbone_dtype(torch.float32)
        ref_model.config.use_cache = False
        ref_model.requires_grad_(False)
        ref_model.eval()

    # ---- 数据集（只用 floodnet）----
    train_ds = FloodnetGRPODataset(
        cfg["train_json"], max_samples=cfg.get("max_samples")
    )

    # ---- TrainingArguments（多卡/累积/日志/续训 全靠它）----
    ta_kwargs = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=float(cfg.get("learning_rate", 1e-6)),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=(cfg.get("dtype", "bfloat16") == "bfloat16"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,  # 点云塔+语言塔，保险
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
    )
    if args.max_steps is not None:
        ta_kwargs["max_steps"] = args.max_steps
    elif cfg.get("max_steps"):
        ta_kwargs["max_steps"] = cfg["max_steps"]

    training_args = TrainingArguments(**ta_kwargs)

    trainer = SpatialLMGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        ref_model=ref_model,
        kl_coef=float(cfg.get("kl_coef", 0.04)),
        clip_eps=float(cfg.get("clip_eps", 0.2)),
        num_iterations=cfg.get("num_iterations", 1),
        num_generations=cfg.get("num_generations", 4),
        num_bins=cfg.get("num_bins", 1280),
        max_new_tokens=cfg.get("max_new_tokens", 64),
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        top_k=cfg.get("top_k", 0),
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
