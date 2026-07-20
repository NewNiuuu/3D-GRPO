# -*- coding: utf-8 -*-
"""
SpatialLM GRPO —— 基于 HuggingFace Trainer，自定义 compute_loss 实现 GRPO。

设计要点：
  - 复用 HF Trainer 的全部基建（多卡 DDP、梯度累积、断点续训、日志）。
  - GRPO 的核心逻辑全部写在 compute_loss 里，共 5 步：
      1) 对一个 batch 的每个 prompt（各自带一份点云）采样 G 段 completion
      2) 用占位 reward 给每段打分
      3) 组内标准化算 advantage
      4) 重算每段的 token log-prob（策略梯度需要）
      5) GRPO loss = -(advantage * logprob)，对 completion token 求平均
  - 点云透传天然：generate / forward 都手动带上 point_clouds，无需框架支持。

先跑通版本刻意从简：单步 on-policy（用采样时的 logprob 近似，不做 PPO clip、
不做 KL 到 reference model）。这些是"提质"项，跑通后可增量加入。
"""
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer

from spatiallm_grpo_utils import load_point_cloud_tensor
from reward import compute_reward


def grpo_collate(batch):
    """把 dataset 的样本按原样打包成 list（点云在 compute_loss 里惰性加载）。"""
    return {
        "pcd_path": [b["pcd_path"] for b in batch],
        "prompt_text": [b["prompt_text"] for b in batch],
        "answer": [b["answer"] for b in batch],
    }


class SpatialLMGRPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        tokenizer=None,
        num_generations: int = 4,
        num_bins: int = 1280,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        gen_batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._tok = tokenizer
        self.num_generations = num_generations
        self.num_bins = num_bins
        self.max_new_tokens = max_new_tokens
        self.gen_temperature = temperature
        self.gen_top_p = top_p
        self.gen_top_k = top_k
        self._pcd_cache = {}

    # 用我们自定义的 collate，保留 pcd_path/prompt_text/answer 三个字段
    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            collate_fn=grpo_collate,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
        )

    def _load_pcd(self, path) -> torch.Tensor:
        """惰性加载 + 缓存点云张量 (N, 9)。"""
        if path not in self._pcd_cache:
            self._pcd_cache[path] = load_point_cloud_tensor(path, self.num_bins)
        return self._pcd_cache[path]

    def _build_prompt_ids(self, prompt_text: str) -> torch.Tensor:
        model = self.model
        base = self.model.module if hasattr(self.model, "module") else self.model
        if base.config.model_type in ("spatiallm_qwen", "spatiallm_qwen3"):
            conv = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ]
        else:
            conv = [{"role": "user", "content": prompt_text}]
        ids = self._tok.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return ids

    @torch.no_grad()
    def _sample(self, base_model, prompt_ids, pcd_batched):
        """对单个 prompt 采样 num_generations 段。返回 full_ids (G, L)。

        采样是纯推理，需要 KV cache 且不需要梯度。若模型开着 gradient
        checkpointing，会与 cache 冲突并每层刷警告——这里临时关掉 gc、
        采样结束后恢复，既消警告又让采样更快。
        """
        gc_was_on = getattr(base_model, "is_gradient_checkpointing", False)
        if gc_was_on:
            base_model.gradient_checkpointing_disable()
        base_model.config.use_cache = True
        # 关键：采样必须在 eval 模式下做。模型 forward 进入"点云处理块"的条件是
        # `input_ids.shape[1] != 1 or self.training`。若保持 training=True，自回归
        # generate 从第二步起每步只喂 1 个 token（shape[1]==1），却因 training=True
        # 仍进入点云块，而该 token 不含 point-token → 断言 "got 0 and 0" 崩溃。
        # 切到 eval 后，只有第一步(喂完整含 point-token 的 prompt)会处理点云，正确。
        was_training = base_model.training
        base_model.eval()
        try:
            out = base_model.generate(
                input_ids=prompt_ids,
                point_clouds=pcd_batched,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.gen_temperature,
                top_p=self.gen_top_p,
                top_k=self.gen_top_k if self.gen_top_k > 0 else None,
                num_return_sequences=self.num_generations,
                use_cache=True,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        finally:
            base_model.config.use_cache = False
            if gc_was_on:
                base_model.gradient_checkpointing_enable()
            if was_training:
                base_model.train()
        return out

    def _seq_logprobs(self, base_model, full_ids, prompt_len, pcd_batched):
        """
        对一条完整序列 (1, L) 重算 completion 段每个 token 的 log-prob。
        返回 (gen_len,) 的 logprob 张量（带梯度）。
        """
        attn = torch.ones_like(full_ids)
        out = base_model(
            input_ids=full_ids,
            point_clouds=pcd_batched,
            attention_mask=attn,
            use_cache=False,
        )
        logits = out.logits  # (1, L_expanded, V) —— 点云插入使序列变长
        gen_len = full_ids.shape[1] - prompt_len
        if gen_len <= 0:
            return None
        gen_logits = logits[0, -gen_len - 1 : -1, :]
        gen_targets = full_ids[0, -gen_len:]
        logp = F.log_softmax(gen_logits.float(), dim=-1)
        tok_logp = logp.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)
        return tok_logp

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.module if hasattr(model, "module") else model
        device = self.args.device

        prompt_texts = inputs["prompt_text"]
        pcd_paths = inputs["pcd_path"]
        answers = inputs["answer"]

        total_loss = 0.0
        total_reward = 0.0
        n_seq = 0

        # 逐个 prompt 处理（每个 prompt 是一个 GRPO group）
        for prompt_text, pcd_path, answer in zip(prompt_texts, pcd_paths, answers):
            pcd = self._load_pcd(pcd_path).to(device)
            pcd_batched = pcd.unsqueeze(0)  # (1, N, 9)
            prompt_ids = self._build_prompt_ids(prompt_text).to(device)
            prompt_len = prompt_ids.shape[1]

            # (1) 采样 G 段
            full_ids = self._sample(base, prompt_ids, pcd_batched)  # (G, L)

            # (2) reward
            texts = self._tok.batch_decode(
                full_ids[:, prompt_len:], skip_special_tokens=True
            )
            rewards = torch.tensor(
                [compute_reward(t, answer) for t in texts],
                device=device,
                dtype=torch.float32,
            )

            # (3) 组内标准化 advantage
            adv = rewards - rewards.mean()
            std = rewards.std()
            if std > 1e-6:
                adv = adv / (std + 1e-6)

            # (4)+(5) 逐段重算 logprob，累加 policy loss
            group_loss = 0.0
            valid = 0
            for g in range(self.num_generations):
                seq = full_ids[g : g + 1]
                tok_logp = self._seq_logprobs(base, seq, prompt_len, pcd_batched)
                if tok_logp is None or tok_logp.numel() == 0:
                    continue
                # GRPO(简化)：-adv * mean(logprob over completion tokens)
                group_loss = group_loss + (-adv[g] * tok_logp.mean())
                valid += 1
            if valid > 0:
                total_loss = total_loss + group_loss / valid
                total_reward += rewards.mean().item()
                n_seq += 1

        if n_seq == 0:
            # 极端兜底：构造一个 0 loss 保持计算图
            loss = torch.zeros(1, device=device, requires_grad=True).sum()
        else:
            loss = total_loss / n_seq
            self.log({"grpo/mean_reward": total_reward / n_seq})

        return (loss, None) if return_outputs else loss
