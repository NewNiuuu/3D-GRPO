# -*- coding: utf-8 -*-
"""
SpatialLM GRPO 训练入口（HF Trainer + 自定义 GRPO loss）。

八卡启动示例（在 spatiallm-grpo 环境）：
    cd /home/aiscuser/nyp/3D-RL
    torchrun --nproc_per_node 8 grpo/train_grpo.py --config grpo/config_aircop.yaml

单卡快速验证：
    CUDA_VISIBLE_DEVICES=0 python grpo/train_grpo.py --config grpo/config_test.yaml --max_steps 3

完整命令（含后台训练与监控）见 grpo/README.md 第八节。
"""
import os
import sys
import argparse
import logging as _logging
from datetime import datetime

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
from blob_stream import build_stream_sampler


def main():
    ap = argparse.ArgumentParser("SpatialLM GRPO trainer")
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument("--max_steps", type=int, default=None, help="覆盖 config 的 max_steps（调试用）")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- wandb ----
    # HF 通过 report_to=wandb 自动接管；这里只把项目名/离线模式提前塞进环境变量，
    # 因为 wandb 在 TrainingArguments 构造时就会读它们。
    #
    # ⚠ 本机（AISC 容器）预置了一整套 WANDB_* 环境变量，是给平台自己的任务用的：
    #     WANDB_RUN_ID=7213779716.62777-72df8069-dba0   ← 固定的 run id
    #     WANDB_PROJECT=vllm-sh-wanli
    #     WANDB_NAME=4x-palisades33-LLM2CLIP-yif-unirun-40Ga100
    #     WANDB_RUN_GROUP / WANDB_NOTES
    # wandb 会自动读 WANDB_RUN_ID，于是**每次启动训练都挂到同一个 run 上**：
    # 曲线是历次启动叠在一起的，而且新点的 step 从 1 重新开始、低于服务端已有的
    # 最大 step，图上看着就是"还是上一轮的曲线、也不实时刷新"。
    # 所以这里必须主动清掉继承来的 run 身份，让每次启动都开新 run。
    if "wandb" in str(cfg.get("report_to", "")):
        # 用 = 而不是 setdefault：否则容器里的 WANDB_PROJECT 会盖掉 config
        os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "spatiallm-grpo")
        # 清掉继承来的 run 身份（除非 config 里显式要求续跑某个 run）
        for k in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP", "WANDB_NOTES"):
            os.environ.pop(k, None)
        if cfg.get("run_name"):
            # run 名自动加时间戳，保证**每次启动在 wandb 里都是一条独立、可区分的条目**。
            # 不加的话每次都叫 "urbanvideo-g8-lr1e6"，面板里一排同名 run，
            # 只能靠创建时间猜是哪一次——这正是之前搞混的原因之一。
            # （只有 rank0 会往 wandb 写，各 rank 时间戳差一分钟也没有影响）
            cfg["run_name"] = "{}-{}".format(
                cfg["run_name"], datetime.now().strftime("%m%d-%H%M")
            )
            os.environ["WANDB_NAME"] = cfg["run_name"]
        else:
            os.environ.pop("WANDB_NAME", None)
        # 想续跑之前的 run 时才填：wandb_run_id: xxxx
        if cfg.get("wandb_run_id"):
            os.environ["WANDB_RUN_ID"] = str(cfg["wandb_run_id"])
            os.environ["WANDB_RESUME"] = "allow"
        if cfg.get("wandb_offline", False):
            os.environ["WANDB_MODE"] = "offline"

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
        seed=cfg.get("seed", 42),
        run_name=cfg.get("run_name"),
        logging_first_step=True,
    )
    # 优化器：默认 adamw_torch（fp32 m+v，1.83B 参数约需 14.6GB）。
    # 显存紧张时在 config 里设 optim: adamw_bnb_8bit，可把优化器状态压到约 3.7GB
    # （需 pip install bitsandbytes）。不设则完全沿用 HF 默认行为。
    if cfg.get("optim"):
        ta_kwargs["optim"] = cfg["optim"]
    if args.max_steps is not None:
        ta_kwargs["max_steps"] = args.max_steps
    elif cfg.get("max_steps"):
        ta_kwargs["max_steps"] = cfg["max_steps"]

    training_args = TrainingArguments(**ta_kwargs)

    # ---- 边下边训（可选）----
    # stream_from_blob: true 时不再要求点云已全量落盘，改为「下一窗口 / 训一窗口 /
    # 删一窗口」。磁盘峰值 ≈ 3×window_files 个文件，而不是整个数据集。
    stream, stream_sampler = build_stream_sampler(
        cfg, train_ds, training_args.world_size, training_args.process_index
    )
    if stream_sampler is not None and training_args.process_index == 0:
        print(
            f"[stream] 已启用边下边训：window={cfg.get('stream_window_files', 32)} 文件/窗口, "
            f"本 rank {len(stream_sampler)} 条样本, "
            f"用完即删={cfg.get('stream_delete_after', True)}",
            flush=True,
        )

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
        logp_batch_size=cfg.get("logp_batch_size", 0),
        pcd_cache_size=cfg.get("pcd_cache_size", 32),
        max_points=cfg.get("max_points", 0),
        stream_sampler=stream_sampler,
    )

    trainer.train()
    if torch.cuda.is_available():
        print(
            f"[mem] rank{training_args.process_index} 峰值 "
            f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB / "
            f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB",
            flush=True,
        )
    trainer.save_model()


if __name__ == "__main__":
    main()
