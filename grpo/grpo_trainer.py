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
from collections import OrderedDict
from contextlib import contextmanager
import functools
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from spatiallm_grpo_utils import load_point_cloud_tensor
from reward import compute_reward


class _StreamEpochCallback(TrainerCallback):
    """每个 epoch 开头给流式 sampler 换种子（文件顺序逐轮不同）。"""

    def __init__(self, sampler):
        self.sampler = sampler

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.sampler.set_epoch(int(state.epoch or 0))


def grpo_collate(batch):
    """把 dataset 的样本按原样打包成 list（点云在 compute_loss 里惰性加载）。"""
    return {
        "idx": [b.get("idx", -1) for b in batch],
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
        logp_batch_size: int = 0,
        pcd_cache_size: int = 32,
        max_points: int = 0,
        stream_sampler=None,
        difficulty_log: bool = False,
        **kwargs,
    ):
        # tokenizer 同时交给 HF Trainer（processing_class），否则 save_model /
        # 每个 checkpoint 都只有 model.safetensors 而没有 tokenizer.json、
        # vocab.json、merges.txt，产出的 ckpt 无法直接 from_pretrained 加载。
        if tokenizer is not None:
            kwargs.setdefault("processing_class", tokenizer)
        super().__init__(*args, **kwargs)
        self._tok = tokenizer
        self.num_generations = num_generations
        self.num_bins = num_bins
        self.max_points = int(max_points or 0)
        self.max_new_tokens = max_new_tokens
        self.gen_temperature = temperature
        self.gen_top_p = top_p
        self.gen_top_k = top_k
        # 点云张量缓存：改成有界 LRU。原来是无界 dict，全量数据集下会把所有
        # 预处理后的点云张量堆在内存里（UrbanVideo 单文件 ~125MB，必爆）。
        # 配合 stream_sampler 的窗口式取样，局部性很好，命中率依然高。
        self._pcd_cache = OrderedDict()
        self._pcd_cache_size = max(int(pcd_cache_size), 1)
        # 边下边训：由 sampler 负责「下一批 → 训一批 → 删一批」，见 blob_stream.py
        self.stream_sampler = stream_sampler
        if stream_sampler is not None:
            # HF Trainer 只会对自带 set_epoch 的 dataloader 调用它；我们返回的是
            # 原生 DataLoader，所以用 callback 在每个 epoch 开头换 shuffle 种子，
            # 否则每一轮的文件顺序完全一样。
            self.add_callback(_StreamEpochCallback(stream_sampler))

        # 一次前向算多少条序列的 log-prob。0 = 全组一次算完（G 条）。
        # 显存不够时调小（例如 2），会自动切成多次前向。
        self.logp_batch_size = logp_batch_size if logp_batch_size > 0 else num_generations
        self._reset_log_accum()

        # ---- GRPO 超参 ----
        self.kl_coef = kl_coef          # β：KL 惩罚系数
        self.clip_eps = clip_eps        # PPO clip 范围 ε（单步 num_iterations=1 时不生效）
        self.num_iterations = num_iterations
        # 累计计数（单调递增，跨 step 保留）：有多少个 micro-batch 产生了非零梯度。
        # frac_nonzero_adv 是每步瞬时比例，这个是"到目前为止累计更新次数"。
        self._updated_microbatches = 0

        # ---- 难度打标 ----
        # 训练本身就要对每个 prompt 采 G 条 rollout 并逐条判对错，这份信息原来只被
        # 聚合成 accuracy 就扔了。打开后额外把**每个 group 的逐题结果**落盘，
        # 于是跑一遍训练顺带得到一份全数据集的难度图谱，供下一轮做难题筛选。
        # 开销：每个 group 一行 json，纯 CPU，无同步，可忽略。
        #
        # 每个 rank 写自己的文件，不加锁——8 个进程往同一文件追加会交错撕裂。
        # 聚合交给 grpo/analyze_difficulty.py。
        self._diff_fp = None
        if difficulty_log:
            d = os.path.join(self.args.output_dir, "difficulty")
            os.makedirs(d, exist_ok=True)
            self._diff_fp = open(
                os.path.join(d, f"rank{self.args.process_index}.jsonl"),
                "a", buffering=1,          # 行缓冲：训练中途看/中途崩都能拿到已写的部分
            )
        # EOS：真正的对话结束符（Qwen 为 <|im_end|>）。completion mask 以此截断，
        # 之后的 pad token 不计入 loss / KL。
        self.eos_token_id = tokenizer.eos_token_id if tokenizer is not None else None

        # ---- reference model（冻结，用于 KL）----
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()
            self.ref_model.config.use_cache = False
            # 立刻搬上卡，别等到 _ref_logprobs 里惰性搬。
            # 惰性搬的时机恰好是策略前向图还挂着的显存最高点，那时再要 3.6GB
            # 常常正好 OOM（UrbanVideo 上实测：module.py convert 处 38.3/39.5GB 炸）。
            # 提前搬不改变总量，但把这笔固定开销挪到峰值之外，失败也失败在启动时。
            self.ref_model.to(self.args.device)

    # 用我们自定义的 collate，保留 pcd_path/prompt_text/answer 三个字段
    def get_train_dataloader(self) -> DataLoader:
        # 边下边训：StreamingGroupSampler 自己就带了 rank 切分（文件轮流发牌）
        # 和跨 rank 步数截齐，所以它直接顶替 DistributedSampler。
        if self.stream_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self._train_batch_size,
                sampler=self.stream_sampler,
                collate_fn=grpo_collate,
                # 流式下 sampler 在主进程按窗口调度下载，worker 只做 collate；
                # 这里保持 0 让 index 的产出节奏和窗口调度严格对齐。
                num_workers=0,
                drop_last=True,
            )
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
        """惰性加载 + LRU 缓存点云张量 (N, 9)。

        边下边训模式下文件可能还没落盘/已被删除，交给 stream_sampler 兜底：
        它知道该文件属于哪个窗口，能按需把它拉回来。
        """
        hit = self._pcd_cache.pop(path, None)
        if hit is not None:
            self._pcd_cache[path] = hit  # 移到 LRU 末尾
            return hit
        if self.stream_sampler is not None:
            self.stream_sampler.stream.ensure_file(path)
        t = load_point_cloud_tensor(path, self.num_bins, max_points=self.max_points)
        self._pcd_cache[path] = t
        while len(self._pcd_cache) > self._pcd_cache_size:
            self._pcd_cache.popitem(last=False)  # 淘汰最久未用
        return t

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

        另外临时打了个 num_logits_to_keep=1 的补丁，见 _keep_last_logits：
        模型自带的 prepare_inputs_for_generation 没传这个参数，prefill 会算
        整条序列的 logits，而 generate 只用最后一个位置。
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
            with self._keep_last_logits(base_model), self._share_point_encoding(base_model):
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

    @contextmanager
    def _keep_last_logits(self, base_model):
        """采样期间让 prefill 只算最后一个位置的 logits。

        SpatialLM 自己重写了 prepare_inputs_for_generation，但没往 model_inputs
        里放 num_logits_to_keep，于是 forward 拿到默认值 0 —— 而它内部写的是
        `hidden_states[:, -num_logits_to_keep:, :]`，`-0` 切片等于**整条序列**。
        点云会把序列展开到几千 token，词表 151936，于是 prefill 白算一个
        (G, L, V) 的巨型 logits，而 generate 只用得到最后一行。

        这里只在 GRPO 的采样路径上临时包一层，不改 spatiallm/ 下与 SFT、
        inference 共用的模型文件。数值上完全等价（generate 取的就是最后一行）。
        """
        orig = base_model.prepare_inputs_for_generation

        # 必须 wraps：generate 的 _validate_model_kwargs 会用 inspect.signature
        # 反射 prepare_inputs_for_generation 的形参来判断哪些 kwarg 是合法的。
        # 裸的 (*a, **kw) 包装会把签名遮住，导致 point_clouds 被判成非法参数。
        @functools.wraps(orig)
        def patched(*a, **kw):
            mi = orig(*a, **kw)
            mi["num_logits_to_keep"] = 1
            return mi

        base_model.prepare_inputs_for_generation = patched
        try:
            yield
        finally:
            base_model.prepare_inputs_for_generation = orig

    @contextmanager
    def _share_point_encoding(self, base_model):
        """让一个 group 内的 G 条序列共用一次点云编码。

        模型 forward 里是逐条编码的：
            for i in range(n_point_clouds): self.forward_point_cloud(point_clouds[i], ...)
        而 GRPO 的一次前向里，这 G 份点云**本来就是同一份**（同一个 prompt 采样出
        的 G 条 rollout）。于是 Sonata 编码器被白跑 G 遍：AirCop 单云 1183 点无所谓，
        UrbanVideo 单云 5 万点、fp32 注意力矩阵 (N', H, K, K)，G=8 直接把 40G A100
        撑爆（实测只加载策略模型、无优化器状态就 OOM）。

        这里在 GRPO 的前向路径上临时 memo：第一条编码，后面 G-1 条直接复用同一个
        张量对象。数值完全等价；反向时梯度在这一次编码上累加，与编 G 遍求和一致。
        只包 grpo/ 的调用路径，不改 spatiallm/ 下与 SFT、inference 共用的模型文件。

        安全前提：本 trainer 的每次模型调用都只对应一个 group（compute_loss 逐个
        prompt 处理，_sample 也是单 prompt + num_return_sequences=G），所以"整个
        batch 共享同一份点云"恒成立。这里额外断言 shape 一致做二次保险。
        """
        inner = base_model
        while hasattr(inner, "module"):  # 剥掉 DDP / 其它 wrapper
            inner = inner.module
        if not hasattr(inner, "forward_point_cloud"):
            yield
            return

        orig = inner.forward_point_cloud
        memo = {}

        @functools.wraps(orig)
        def patched(point_cloud, device, dtype):
            key = (tuple(point_cloud.shape), str(device), dtype)
            if key not in memo:
                memo[key] = orig(point_cloud, device, dtype)
            return memo[key]

        inner.forward_point_cloud = patched
        try:
            yield
        finally:
            inner.forward_point_cloud = orig
            memo.clear()

    def _seq_logprobs(self, base_model, full_ids, prompt_len, pcd_batched):
        """
        重算 completion 段每个 token 的 log-prob。返回 (B, gen_len)（带梯度）。

        支持 B>1 批处理：同一个 group 里的 G 条序列共享同一个 prompt 和同一份
        点云，generate 也把它们 pad 到了同样长度，因此点云 token 的展开量完全
        一致 → 尾部对齐索引对每一行都成立，可以安全地一次前向算完。
        （原实现逐条 batch=1 跑，G=4 时策略+ref 要 8 次前向；批处理后只要 2 次。）

        右 padding + 因果注意力下，pad 位置排在真实 token 之后，不会影响前面
        token 的 logits，所以 attention_mask 全 1 与精确 mask 数值等价。
        """
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)
        B = full_ids.shape[0]
        # 点云按 batch 复制（同 group 内是同一份点云）。
        # 用 expand 不做 contiguous：真正的 G 份副本没必要存在，
        # _share_point_encoding 会保证只有第 0 行被编码器读到。
        if pcd_batched.shape[0] != B:
            pcd_batched = pcd_batched.expand(B, -1, -1)

        attn = torch.ones_like(full_ids)
        gen_len = full_ids.shape[1] - prompt_len
        if gen_len <= 0:
            return None
        # 只要末尾 gen_len+1 个位置的 logits（多要 1 个是因为预测第 t 个 token
        # 用的是第 t-1 个位置）。不设的话模型默认算整条展开后序列的 logits，
        # 点云展开几千 token × 词表 151936，白白吃掉好几 GB 且要进反向图。
        with self._share_point_encoding(base_model):
            out = base_model(
                input_ids=full_ids,
                point_clouds=pcd_batched,
                attention_mask=attn,
                use_cache=False,
                num_logits_to_keep=gen_len + 1,
            )
        gen_logits = out.logits[:, :-1, :]              # (B, gen_len, V)
        gen_targets = full_ids[:, -gen_len:]            # (B, gen_len)
        # 只对 (B, gen_len, V) 做 float32（gen_len<=max_new_tokens，很小）
        logp = F.log_softmax(gen_logits.float(), dim=-1)
        tok_logp = logp.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)
        return tok_logp  # (B, gen_len)

    def _logprobs_chunked(self, base_model, full_ids, prompt_len, pcd, chunk):
        """按 chunk 大小分批算 log-prob，拼回 (G, gen_len)。chunk 用于控显存。

        共享点云编码的上下文开在 chunk 循环**外面**：否则每个 chunk 各编一次云，
        UrbanVideo 那种 5 万点的云白跑 ceil(G/chunk) 遍。开在外面时所有 chunk
        复用同一份编码结果（梯度图本来也要留到 backward，不额外占显存）。
        """
        G = full_ids.shape[0]
        with self._share_point_encoding(base_model):
            if chunk >= G:
                return self._seq_logprobs(base_model, full_ids, prompt_len, pcd)
            outs = []
            for i in range(0, G, chunk):
                outs.append(
                    self._seq_logprobs(
                        base_model, full_ids[i : i + chunk], prompt_len, pcd
                    )
                )
            return torch.cat(outs, dim=0)

    def _completion_mask(self, gen_targets: torch.Tensor) -> torch.Tensor:
        """
        为 completion 段构造 mask：保留到第一个 EOS（含），之后的 pad token 置 0。

        generate 采样时短回答会提前生成 EOS(<|im_end|>) 再用 pad_token 填充；
        那些 pad 不是真实输出，不应参与 policy loss / KL。若整段没有 EOS
        （生成到 max_new_tokens 截断），则全部保留。

        支持 (gen_len,) 与 (B, gen_len) 两种输入，返回同形状 float mask。
        """
        squeeze_back = gen_targets.dim() == 1
        t = gen_targets.unsqueeze(0) if squeeze_back else gen_targets
        if self.eos_token_id is None:
            mask = torch.ones_like(t, dtype=torch.float32)
        else:
            is_eos = (t == self.eos_token_id)
            # 当前位置之前（不含自身）出现过的 EOS 个数；==0 即"首个 EOS 及之前"
            prior_eos = is_eos.cumsum(dim=1) - is_eos.long()
            mask = (prior_eos == 0).to(torch.float32)
        return mask.squeeze(0) if squeeze_back else mask

    @torch.no_grad()
    def _ref_logprobs(self, full_ids, prompt_len, pcd_batched, device, chunk=None):
        """用冻结 ref model 重算 completion logprob（无梯度），用于 KL。

        ref 与 policy 结构一致、点云插入方式一致，故尾部对齐索引与
        _seq_logprobs 相同。返回 (B, gen_len) 或 None。
        """
        if self.ref_model is None:
            return None
        # 惰性把 ref 挪到与 policy 同 device（Trainer 只搬 self.model）。
        if next(self.ref_model.parameters()).device != device:
            self.ref_model.to(device)
        return self._logprobs_chunked(
            self.ref_model, full_ids, prompt_len, pcd_batched,
            chunk or self.logp_batch_size,
        )

    def _flush_logs(self, force=False):
        """把累计的诊断统计按「每个 optimizer step 一次」的节奏吐给 HF/wandb。

        原实现在 compute_loss 里直接 self.log()，accum=4 时一个 optimizer step
        会打 4 条瞬时值，HF 不做累积平均 → wandb 曲线锯齿严重、且与 loss/
        grad_norm 的 step 轴对不齐。这里改成累积到一个 optimizer step 边界
        再取平均发一次。
        """
        acc = self._log_accum
        n = acc["n_micro"]
        if n == 0:
            return
        ga = max(int(self.args.gradient_accumulation_steps), 1)
        if not force and n < ga:
            return
        seen = max(acc["n_groups_seen"], 1)
        upd = max(acc["n_groups_upd"], 1)
        self.log(
            {
                "grpo/mean_reward": acc["reward"] / seen,
                "grpo/accuracy": acc["acc"] / seen,
                "grpo/reward_std": acc["reward_std"] / seen,
                "grpo/kl": acc["kl"] / upd,
                "grpo/completion_len": acc["comp_len"] / upd,
                "grpo/frac_nonzero_adv": acc["n_groups_upd"] / seen,
                # 累计：到目前为止有多少个 micro-batch 实际吃到非零梯度（单调递增）
                "grpo/updated_microbatches": float(self._updated_microbatches),
                "grpo/skipped_groups": float(acc["n_groups_seen"] - acc["n_groups_upd"]),
                # 本卡显存（GB）。mem_gb 是**上一次日志以来的步内峰值**，不是步末
                # 当前值——OOM 由瞬时峰值决定，而步末采样看不见它。实测教训：
                # 8 卡上 memory_allocated() 稳在 19.65GB 的"平坦"曲线底下，藏着
                # 能冲到 39.5GB 的尖峰，照着平坦曲线判断"还很宽松"会直接踩空。
                # 这条曲线必须又平又低；持续上涨=泄漏，忽高忽低=有长尾样本。
                # mem_now 保留当前值，两条一起看才能区分"常驻涨"和"瞬时尖峰"。
                "grpo/mem_gb": torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0,
                "grpo/mem_now_gb": torch.cuda.memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0,
            }
        )
        if torch.cuda.is_available():
            # 清零高水位，让下一条 mem_gb 反映的是下一个区间的峰值而不是历史最大
            torch.cuda.reset_peak_memory_stats()
        self._reset_log_accum()

    def _wrap_model(self, *args, **kwargs):
        """在 HF 建好 DDP 配置之后，补一个它没暴露的省显存开关。

        默认 DDP 同时持有「梯度本体」和「allreduce 通信桶」两份完整拷贝，
        1.83B 参数 bf16 每份 3.7GB —— 多卡因此比单卡凭空多吃约 7.4GB。
        实测 UrbanVideo：单卡峰值 28.3GB，4 卡 36.0GB，差值 7.7GB 正好对上。
        gradient_as_bucket_view 让 param.grad 指向桶内偏移，省掉其中一份。

        为什么必须在这里改而不是 __init__：`accelerator.ddp_handler` 是
        Trainer._wrap_model() 里现场 new 出来的（transformers/trainer.py 约 2097 行），
        __init__ 时还不存在，而且就算提前塞了也会被这里整个覆盖掉。
        真正读取它的是随后的 accelerator.prepare()，所以此刻改还来得及。
        """
        model = super()._wrap_model(*args, **kwargs)
        handler = getattr(self.accelerator, "ddp_handler", None)
        if handler is not None:
            handler.gradient_as_bucket_view = True
        return model

    def _reset_log_accum(self):
        self._log_accum = {
            "n_micro": 0, "n_groups_seen": 0, "n_groups_upd": 0,
            "reward": 0.0, "acc": 0.0, "reward_std": 0.0,
            "kl": 0.0, "comp_len": 0.0,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.module if hasattr(model, "module") else model
        device = self.args.device

        prompt_texts = inputs["prompt_text"]
        pcd_paths = inputs["pcd_path"]
        answers = inputs["answer"]
        idxs = inputs.get("idx") or [-1] * len(prompt_texts)

        # 按「总有效 token 数」归一（DAPO/Dr.GRPO 式）：累加所有 group / 段 / 有效
        # token 的 loss 项，最后除以有效 token 总数。CoT 变长时每 token 权重一致。
        loss_sum = torch.zeros((), device=device)
        tok_total = 0
        acc = self._log_accum

        # 逐个 prompt 处理（每个 prompt 是一个 GRPO group）
        for prompt_text, pcd_path, answer, item_idx in zip(
            prompt_texts, pcd_paths, answers, idxs
        ):
            acc["n_groups_seen"] += 1
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
            acc["reward"] += rewards.mean().item()
            acc["reward_std"] += std.item()
            acc["acc"] += (rewards == 1.0).float().mean().item()

            # (3a) 难度打标：必须写在下面 continue 之前。
            # 被 std<=1e-6 跳过的组正是「G 条全对」和「G 条全错」两类——
            # 也就是最该被标记的"太简单"和"太难"，写在 continue 之后就全丢了。
            if self._diff_fp is not None:
                n_ok = int((rewards == 1.0).sum().item())
                self._diff_fp.write(
                    json.dumps(
                        {
                            "idx": int(item_idx),
                            "step": int(self.state.global_step),
                            "epoch": round(float(self.state.epoch or 0.0), 3),
                            "n_correct": n_ok,
                            "n_gen": int(rewards.numel()),
                            "gt": answer,
                            # 采到的答案分布：看错的时候是错成同一个选项(系统性偏差)
                            # 还是散着错(纯不会)，两者要的干预完全不同
                            "preds": [t.strip()[:8] for t in texts],
                            "pcd": pcd_path,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # advantage 全零（组内 reward 无差异）→ 无学习信号，跳过 forward 省算力
            if std <= 1e-6:
                continue
            adv = (rewards - rewards.mean()) / (std + 1e-6)  # (G,)

            gen_len = full_ids.shape[1] - prompt_len
            if gen_len <= 0:
                continue
            gen_targets = full_ids[:, prompt_len:]              # (G, gen_len)
            mask = self._completion_mask(gen_targets)            # (G, gen_len)
            n_valid = mask.sum()
            if n_valid < 1:
                continue

            # ---- (4a) 先算冻结 ref 的 logprob ----
            # 必须在 policy 带梯度前向**之前**算。ref 是 no_grad 的，算完只留下
            # (G, gen_len) 这么点结果，中间激活立刻释放；而如果放在 policy 之后，
            # ref 的瞬时激活会叠在 policy 那张还没 backward 的反向图上面，
            # 两个峰值重叠。数学上完全等价——ref 冻结，ref_logp 不参与求导。
            ref_logp = None
            if self.ref_model is not None:
                ref_logp = self._ref_logprobs(
                    full_ids, prompt_len, pcd_batched, device
                )

            # (4)+(5) 一次性重算 G 条的 logprob + KL（批处理，不再逐条循环）
            tok_logp = self._logprobs_chunked(
                base, full_ids, prompt_len, pcd_batched, self.logp_batch_size
            )  # (G, gen_len)
            if tok_logp is None:
                continue

            # ---- policy 项：-min(ratio·A, clip(ratio)·A) ----
            # 单步（num_iterations=1）：old logprob = 当前 logprob 的 detach，
            # ratio 恒=1，clip 不触发。写成标准形式为多步预留接口。
            old_logp = tok_logp.detach()
            ratio = torch.exp(tok_logp - old_logp)
            a = adv.unsqueeze(1)                                 # (G, 1) 广播到每个 token
            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a
            policy_tok = -torch.min(unclipped, clipped)          # (G, gen_len)

            # ---- KL 项（k3 无偏估计，逐 token ≥0）----
            # ref_logp 已在 policy 前向之前算好（见上面 4a），这里只做纯张量运算
            if ref_logp is not None:
                diff = ref_logp - tok_logp  # logπ_ref - logπ_θ
                kl_tok = torch.exp(diff) - diff - 1.0
            else:
                kl_tok = torch.zeros_like(tok_logp)

            per_tok = policy_tok + self.kl_coef * kl_tok         # (G, gen_len)
            loss_sum = loss_sum + (per_tok * mask).sum()
            tok_total += int(n_valid.item())

            # 诊断统计：每条序列先按自身有效 token 求均值，再对 G 条取平均
            per_seq_valid = mask.sum(dim=1).clamp(min=1)         # (G,)
            acc["kl"] += float(((kl_tok * mask).sum(dim=1) / per_seq_valid).mean().item())
            acc["comp_len"] += float(mask.sum(dim=1).float().mean().item())
            acc["n_groups_upd"] += 1

        if tok_total == 0:
            # 极端兜底：本 batch 所有 group 都无学习信号，构造 0 loss 保持计算图
            loss = torch.zeros(1, device=device, requires_grad=True).sum()
        else:
            # 本 micro-batch 产生了非零梯度 → 累计更新计数 +1
            self._updated_microbatches += 1
            loss = loss_sum / tok_total

        # 无论本 micro-batch 是否有梯度都要计数，否则 frac_nonzero_adv 会偏高
        acc["n_micro"] += 1
        self._flush_logs()

        return (loss, None) if return_outputs else loss
