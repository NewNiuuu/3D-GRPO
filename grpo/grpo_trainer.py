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

先跑通版本刻意从简：单步 on-policy（num_iterations=1，采样策略=更新策略，ratio 恒=1，
clip 不触发）。已加入 completion mask（EOS 后 pad 不计入 loss）、reference model 的
KL 惩罚（k3 无偏估计）、按总有效 token 归一（为 CoT 预留）。多步更新（复用 rollout
做多次梯度更新）尚未实现，届时把 num_iterations 调大并缓存 old logprob 即可启用 clip。
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
        ref_model=None,
        kl_coef: float = 0.04,
        clip_eps: float = 0.2,
        num_iterations: int = 1,
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

        # ---- GRPO 超参 ----
        self.kl_coef = kl_coef          # β：KL 惩罚系数
        self.clip_eps = clip_eps        # PPO clip 范围 ε（单步 num_iterations=1 时不生效）
        self.num_iterations = num_iterations
        # 累计计数（单调递增，跨 step 保留）：有多少个 micro-batch 产生了非零梯度。
        # frac_nonzero_adv 是每步瞬时比例，这个是"到目前为止累计更新次数"。
        self._updated_microbatches = 0
        # EOS：真正的对话结束符（Qwen 为 <|im_end|>）。completion mask 以此截断，
        # 之后的 pad token 不计入 loss / KL。
        self.eos_token_id = tokenizer.eos_token_id if tokenizer is not None else None

        # ---- reference model（冻结，用于 KL）----
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()
            self.ref_model.config.use_cache = False

    # 用我们自定义的 collate，保留 pcd_path/prompt_text/answer 三个字段
    def get_train_dataloader(self) -> DataLoader:
        # 多卡：必须用 DistributedSampler 把数据切分到各 rank，否则每张卡都遍历
        # 全量数据 → 数据严重重叠、梯度不是预期的有效 batch。单卡则普通随机打乱。
        if self.args.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=True,
                seed=self.args.seed,
                drop_last=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self._train_batch_size,
                sampler=sampler,  # 有 sampler 时不能再传 shuffle
                collate_fn=grpo_collate,
                num_workers=self.args.dataloader_num_workers,
                drop_last=True,
            )
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

    def _completion_mask(self, gen_targets: torch.Tensor) -> torch.Tensor:
        """
        为 completion 段构造 mask：保留到第一个 EOS（含），之后的 pad token 置 0。

        generate 采样时短回答会提前生成 EOS(<|im_end|>) 再用 pad_token 填充；
        那些 pad 不是真实输出，不应参与 policy loss / KL。若整段没有 EOS
        （生成到 max_new_tokens 截断），则全部保留。
        返回 (gen_len,) 的 float mask（1/0），与 gen_targets 同 device。
        """
        gen_len = gen_targets.shape[0]
        mask = torch.ones(gen_len, device=gen_targets.device, dtype=torch.float32)
        if self.eos_token_id is None:
            return mask
        eos_pos = (gen_targets == self.eos_token_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            first = eos_pos[0, 0].item()
            if first + 1 < gen_len:
                mask[first + 1 :] = 0.0
        return mask

    @torch.no_grad()
    def _ref_logprobs(self, full_ids, prompt_len, pcd_batched, device):
        """用冻结 ref model 重算 completion logprob（无梯度），用于 KL。

        ref 与 policy 结构一致、点云插入方式一致，故尾部对齐索引与
        _seq_logprobs 相同。返回 (gen_len,) 或 None。
        """
        if self.ref_model is None:
            return None
        # 惰性把 ref 挪到与 policy 同 device（Trainer 只搬 self.model）。
        if next(self.ref_model.parameters()).device != device:
            self.ref_model.to(device)
        return self._seq_logprobs(self.ref_model, full_ids, prompt_len, pcd_batched)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.module if hasattr(model, "module") else model
        device = self.args.device

        prompt_texts = inputs["prompt_text"]
        pcd_paths = inputs["pcd_path"]
        answers = inputs["answer"]

        # 按「总有效 token 数」归一（DAPO/Dr.GRPO 式）：累加所有 group / 段 / 有效
        # token 的 loss 项，最后除以有效 token 总数。CoT 变长时每 token 权重一致。
        loss_sum = torch.zeros((), device=device)
        tok_total = 0
        # 诊断统计
        total_reward = 0.0
        total_reward_std = 0.0
        total_acc = 0.0        # 选择题答对率（reward==1 的比例）；0/1 reward 下 == mean_reward
        total_kl = 0.0
        total_comp_len = 0.0
        n_groups = 0          # 参与更新的 group 数（advantage 非零）
        n_groups_seen = 0     # 见到的 group 数（含被跳过的）

        # 逐个 prompt 处理（每个 prompt 是一个 GRPO group）
        for prompt_text, pcd_path, answer in zip(prompt_texts, pcd_paths, answers):
            n_groups_seen += 1
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
            std = rewards.std()
            total_reward += rewards.mean().item()
            total_reward_std += std.item()
            total_acc += (rewards == 1.0).float().mean().item()
            # advantage 全零（组内 reward 无差异）→ 无学习信号，跳过 forward 省算力
            if std <= 1e-6:
                continue
            adv = (rewards - rewards.mean()) / (std + 1e-6)

            # (4)+(5) 逐段重算 logprob + KL，按标准 GRPO 形式累加
            group_kl = 0.0
            group_comp_len = 0.0
            group_valid_seq = 0
            for g in range(self.num_generations):
                seq = full_ids[g : g + 1]
                tok_logp = self._seq_logprobs(base, seq, prompt_len, pcd_batched)
                if tok_logp is None or tok_logp.numel() == 0:
                    continue
                gen_targets = seq[0, prompt_len:]
                mask = self._completion_mask(gen_targets)  # (gen_len,)
                n_valid = mask.sum()
                if n_valid < 1:
                    continue

                # ---- policy 项：-min(ratio·A, clip(ratio)·A) ----
                # 单步（num_iterations=1）：old logprob = 当前 logprob 的 detach，
                # ratio 恒=1，clip 不触发。写成标准形式为多步预留接口。
                old_logp = tok_logp.detach()
                ratio = torch.exp(tok_logp - old_logp)
                a = adv[g]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a
                policy_tok = -torch.min(unclipped, clipped)  # (gen_len,)

                # ---- KL 项（k3 无偏估计，逐 token ≥0）----
                if self.ref_model is not None:
                    ref_logp = self._ref_logprobs(seq, prompt_len, pcd_batched, device)
                    diff = ref_logp - tok_logp  # logπ_ref - logπ_θ
                    kl_tok = torch.exp(diff) - diff - 1.0
                else:
                    kl_tok = torch.zeros_like(tok_logp)

                per_tok = policy_tok + self.kl_coef * kl_tok  # (gen_len,)
                # 累加有效 token 的 loss（按总 token 归一，最后统一除）
                loss_sum = loss_sum + (per_tok * mask).sum()
                tok_total += int(n_valid.item())

                group_kl += float((kl_tok * mask).sum().item() / max(n_valid.item(), 1))
                group_comp_len += float(n_valid.item())
                group_valid_seq += 1

            if group_valid_seq > 0:
                n_groups += 1
                total_kl += group_kl / group_valid_seq
                total_comp_len += group_comp_len / group_valid_seq

        if tok_total == 0:
            # 极端兜底：本 batch 所有 group 都无学习信号，构造 0 loss 保持计算图
            loss = torch.zeros(1, device=device, requires_grad=True).sum()
        else:
            # 本 micro-batch 产生了非零梯度 → 累计更新计数 +1
            self._updated_microbatches += 1
            loss = loss_sum / tok_total
            denom = max(n_groups_seen, 1)
            self.log(
                {
                    "grpo/mean_reward": total_reward / denom,
                    "grpo/accuracy": total_acc / denom,
                    "grpo/reward_std": total_reward_std / denom,
                    "grpo/kl": (total_kl / n_groups) if n_groups > 0 else 0.0,
                    "grpo/completion_len": (total_comp_len / n_groups) if n_groups > 0 else 0.0,
                    "grpo/frac_nonzero_adv": n_groups / denom,
                    # 累计：到目前为止有多少个 micro-batch 实际吃到非零梯度（单调递增）
                    "grpo/updated_microbatches": float(self._updated_microbatches),
                }
            )

        return (loss, None) if return_outputs else loss
