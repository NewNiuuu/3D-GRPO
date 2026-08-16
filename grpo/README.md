# SpatialLM GRPO —— 运行可行性调查报告

> 调查日期：2026-08-15，环境就绪与实跑验证：2026-08-16
> 运行环境：本机 `/home/aiscuser/nyp/3D-RL`，8× A100-SXM4-40GB
> 结论：**已在本机跑通**。单卡 3 步、四卡 6 步冒烟训练均通过，产出 ckpt 可独立加载。详见[第七节](#七本机就绪状态2026-08-16-实测)。

---

## 目录

1. [入口与调用链](#一入口与调用链)
2. [参数从哪里传入](#二参数从哪里传入三层)
3. [超参清单与含义](#三超参清单与含义)
4. [一个 step 的完整数据流](#四一个-step-的完整数据流)
5. [预期输出](#五预期输出)
6. [监控方式](#六监控方式)
7. [本机就绪状态](#七本机就绪状态2026-08-16-实测)
8. [建议的启动顺序](#八建议的启动顺序)

---

## 一、入口与调用链

**入口：`grpo/train_grpo.py`**，唯一必需参数是 `--config`。

```
train_grpo.py  main()
 ├─ yaml.safe_load(config)                          # 读超参
 ├─ AutoTokenizer.from_pretrained(model_path)
 ├─ load_spatiallm(model_path)          → policy π_θ  （可训练）
 │    └─ set_point_backbone_dtype(fp32) # 点云塔保持 fp32，与推理一致
 ├─ load_spatiallm(model_path)          → ref π_ref   （冻结，kl_coef>0 才加载）
 ├─ FloodnetGRPODataset(train_json)     → {pcd_path, prompt_text, answer}
 ├─ TrainingArguments(...)              # 多卡/累积/日志/存档 全交给 HF
 ├─ SpatialLMGRPOTrainer(...)           # GRPO 逻辑在 compute_loss
 ├─ trainer.train()
 └─ trainer.save_model()                # 训完存到 output_dir
```

启动命令（代码 docstring 里给的）：

```bash
# 多卡
torchrun --nproc_per_node 4 grpo/train_grpo.py --config grpo/config_test.yaml

# 单卡调试（只跑 3 步）
CUDA_VISIBLE_DEVICES=0 python grpo/train_grpo.py --config grpo/config_test.yaml --max_steps 3
```

> ⚠️ 必须**在仓库根目录 `3D-RL/` 下运行**，因为 config 里的 `train_json: data/grpo_test.json` 是相对路径。

---

## 二、参数从哪里传入（三层）

### 第 1 层：YAML —— 唯一需要改的地方

`grpo/config_test.yaml` 是**全部超参的单一来源**。目前仓库里只有这一个 GRPO config。

### 第 2 层：命令行 —— 只有一个 override

`train_grpo.py:42` 只暴露了 `--max_steps`，优先级高于 YAML（`train_grpo.py:90-93`）。其余参数改不了，只能改 YAML。

### 第 3 层：分流到两个对象

YAML 的键在 `main()` 里被拆成两组：

| 去向 | 参数 | 代码位置 |
|---|---|---|
| **`TrainingArguments`**（HF 管） | `output_dir, per_device_train_batch_size, gradient_accumulation_steps, learning_rate, num_train_epochs, logging_steps, save_steps, save_total_limit, dtype→bf16, gradient_checkpointing, dataloader_num_workers, report_to, warmup_ratio, lr_scheduler_type` | `train_grpo.py:72-95` |
| **`SpatialLMGRPOTrainer` kwargs**（GRPO 自己管） | `kl_coef, clip_eps, num_iterations, num_generations, num_bins, max_new_tokens, temperature, top_p, top_k` | `train_grpo.py:97-112` |

另有两个**写死在代码里、YAML 改不了**的：

- `remove_unused_columns=False` —— 必须，否则 HF 会把 `pcd_path/prompt_text/answer` 三个非标准字段删掉
- `ddp_find_unused_parameters=True` —— 点云塔 + 语言塔，保险起见

---

## 三、超参清单与含义

`grpo/config_test.yaml` 全文 42 行，分 5 组：

```yaml
### 模型
model_path: /root/lnj/saves/point_mixed_downsample   # ← 必改：SFT ckpt 路径
dtype: bfloat16

### 数据
train_json: data/grpo_test.json    # 相对仓库根目录
max_samples: null                  # 调试设 32，比 --max_steps 更省启动时间
num_bins: 1280                     # 点云离散化网格数，必须与 SFT 一致

### GRPO 采样
num_generations: 4                 # G，组大小。太小 advantage 噪声大，太大显存/时间线性涨
max_new_tokens: 64                 # 选择题答案短，够用
temperature: 1.0 / top_p: 1.0 / top_k: 0    # 纯采样，不截断分布 → 保证 on-policy 无偏

### GRPO 目标
kl_coef: 0.04       # β。设 0 可省掉一整份 ref model 显存（约 3.4GB）
clip_eps: 0.2       # num_iterations=1 时不生效
num_iterations: 1   # 单步 on-policy；>1 的多步更新尚未实现，调大无效

### 训练
per_device_train_batch_size: 1     # 每卡每次 1 个 prompt = 1 个 group
gradient_accumulation_steps: 4
learning_rate: 1.0e-6              # RL 阶段必须比 SFT(2e-5) 小一到两个量级
lr_scheduler_type: constant
gradient_checkpointing: true
```

**有效 batch** = 卡数 × `per_device` × `accum`。4 卡时 = 16 个 prompt/step = 64 条 rollout。13578 条数据 ÷ 16 ≈ **849 个 optimizer step/epoch**。

调参优先级建议：`learning_rate` > `num_generations` > `kl_coef`。1e-6 已经很保守，可以先照跑。

---

## 四、一个 step 的完整数据流

```
DataLoader (自定义 grpo_collate，只打包 3 个字符串字段)
   ↓
compute_loss:  对 batch 里每个 prompt（= 一个 group）
   ↓
_load_pcd(path)  → 惰性从 blob/本地读 .ply → (N,9) 张量 → 存进 _pcd_cache
   ↓
_build_prompt_ids  → apply_chat_template(system+user, add_generation_prompt=True)
   ↓
_sample  → generate(G=4 条, 临时 eval() + 临时关 gradient checkpointing)
   ↓
batch_decode → compute_reward(text, "A") → rewards (4,) 全 0/1
   ↓
std<=1e-6 ?  ── 是 → continue（整组跳过，不做任何 forward）
   ↓ 否
adv = (r - mean)/std
   ↓
逐条 g: _seq_logprobs(policy) + _ref_logprobs(ref) + _completion_mask
   ↓
per_tok = -min(ρA, clip(ρ)A) + β·(e^Δ-Δ-1)
   ↓
loss = Σ(per_tok·mask) / Σ mask     ← 按总有效 token 归一
   ↓
HF Trainer 接管：backward → 累积 4 步 → optimizer.step()
```

---

## 五、预期输出

### 产物文件（`output_dir`）

```
saves/grpo_floodnet/
├── checkpoint-500/          # save_steps=500，849 步/epoch 会存 1 个
├── config.json
├── model.safetensors        # 训完 save_model() 存的全量权重（非 LoRA，约 3.4GB）
├── tokenizer.json / ...
└── trainer_state.json       # 完整日志历史，事后画曲线用这个
```

### 控制台日志

会看到两类交替出现的字典。HF 标准的：

```
{'loss': 0.0012, 'grad_norm': 0.83, 'learning_rate': 1e-06, 'epoch': 0.01}
```

以及 `grpo_trainer.py:344-355` 自定义的（每个产生梯度的 micro-batch 打一次）：

```
{'grpo/mean_reward': 0.5,  'grpo/accuracy': 0.5,   'grpo/reward_std': 0.577,
 'grpo/kl': 0.0,           'grpo/completion_len': 3.0,
 'grpo/frac_nonzero_adv': 1.0, 'grpo/updated_microbatches': 7.0}
```

### 各字段该期待什么值

| 字段 | 期望 | 说明 |
|---|---|---|
| `grpo/accuracy` | 起点 = SFT 在这批题上的准确率；**应缓慢上升** | **唯一真正的效果指标** |
| `grpo/frac_nonzero_adv` | 0.3~0.7 | 有多少组产生了学习信号。若 <0.1 说明题目太简单或太难，GRPO 学不动 |
| `grpo/kl` | 从 0 缓慢上升 | 若飙升（>0.5）说明策略跑偏，该调小 lr 或调大 kl_coef |
| `grpo/completion_len` | 2~5 | 选择题答案就几个 token。若接近 64 说明模型在胡扯没吐 EOS |
| `grpo/reward_std` | 0/1 reward + G=4 下只能取 {0, 0.5, 0.577} | 全对/全错时为 0 |
| `loss` | **≈ 0，且没有意义** | 见下方警告 |

> ⚠️ **别看 loss 曲线。** `num_iterations=1` 时 ratio ≡ 1，per-token loss = −A_g；而组内 advantage 之和恒为 0，所以 `loss_sum` 天然接近 0（只有各条长度不等时才有微小偏离）。**loss 不降不代表没在学**，GRPO 的 loss 是个 surrogate，请只盯 `grpo/accuracy` 和 `grpo/kl`。

---

## 六、监控方式（wandb）

`config_aircop.yaml` / `config_urbanvideo.yaml` 已默认 `report_to: wandb`。

```yaml
report_to: wandb
wandb_project: spatiallm-grpo     # 项目名（train_grpo.py 会写进 WANDB_PROJECT）
run_name: aircop-g8-lr1e6         # 这次 run 的名字，wandb 里就显示它
# wandb_offline: true             # 机器连不上外网时改这个，事后 wandb sync 上传
```

首次使用要先登录一次（只需一次，token 存进 `~/.netrc`）：

```bash
/home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/wandb login
```

自定义的 `grpo/*` 键走 `self.log()`，会自动进 wandb，不需要额外改代码。
事后画曲线也可以直接读 `output_dir/trainer_state.json` 里的 `log_history`。

**wandb 上该盯哪几条曲线**（按重要性排序）：

| 面板 | 期望走势 | 不对劲时怎么办 |
|---|---|---|
| `grpo/accuracy` | **缓慢上升**——唯一真正的效果指标 | 一直平 → 看 `frac_nonzero_adv` 是不是太低 |
| `grpo/frac_nonzero_adv` | 0.3~0.7 | <0.1 说明绝大多数组被跳过，等于空转：调大 `num_generations` 或 `temperature` |
| `grpo/kl` | 从 0 缓慢上升 | 飙到 >0.5 说明策略跑偏：调小 `learning_rate` 或调大 `kl_coef` |
| `grpo/completion_len` | 2 左右（1 个字母 + EOS） | 逼近 `max_new_tokens` 说明模型在胡扯没吐 EOS |
| `train/grad_norm` | 有值、不爆炸 | 恒为 0 说明这一步所有组都被跳过了 |
| `train/loss` | **忽略它** | 见下方警告 |

> ⚠️ **别看 loss 曲线。** 已实测 `train_loss = 3.3e-06`。`num_iterations=1` 时 ratio ≡ 1，per-token loss = −A_g，而组内 advantage 之和恒为 0，所以 loss 天然≈0。**loss 不降不代表没在学。**

**日志节奏已修好**：原实现在 `compute_loss` 里直接 `self.log()`，`accum=4` 时一个
optimizer step 打 4 条瞬时值、曲线锯齿严重且与 `loss`/`grad_norm` 的 step 轴对不齐。
现在改成累积到 optimizer step 边界再取平均发一次（`_flush_logs`），
并新增 `grpo/skipped_groups`（本步被跳过的组数）。

---

## 七、本机就绪状态（2026-08-16 实测）

原报告列的 3 个硬阻塞已全部解除，以下为实测结果。

### ① Python 环境 ✅

conda 环境 **`spatiallm-grpo`**（`~/miniconda3/envs/spatiallm-grpo`）：

```
python 3.11 | torch 2.7.1+cu126 | transformers 4.53.0
spconv-cu126 / torch-scatter / timm / open3d / addict 均导入通过
```

原 `vlaser` 环境的 transformers 4.48 没有 Qwen3，已弃用。
`torchsparse` 与 `flash-attn` **不需要**——两者都在 `try/except` 里，本 ckpt 走
SONATA 分支（`spatiallm_qwen3.py:132`）。

### ② 模型权重 ✅

| | |
|---|---|
| blob | `lnj/points/point_mixed_downsample_0720`（相对 `output/liyan`） |
| 本地 | `/home/aiscuser/nyp/ckpts/point_mixed_downsample` |

17 个文件全部下载并逐个核对字节数，6.83 GiB，1.830 B 参数。

> ⚠️ 用 `blob_manager.sh` 下载时，路径要写**相对 `/output/liyan` 的路径**，
> 工具会自动拼 `BASE_PREFIX`。写成 `/blob/output/liyan/...` 会被拼成
> `output/liyan/blob/output/liyan/...`，报 `The specified file was not found`。

### ③ 点云数据 ✅

| | |
|---|---|
| blob | `Pointcloud-VQA/AirCopBench/{Real2,Sim3,Sim5,Sim6}_VQA_train` |
| 本地 | `/home/aiscuser/nyp/pcdata/Pointcloud-VQA/AirCopBench/...` |
| 符号链接 | `/Pointcloud-VQA -> /home/aiscuser/nyp/pcdata/Pointcloud-VQA` |

935 个 `.ply`，5.41 GiB，与 `grpo_test.json` 引用的唯一点云数**完全一致**。
靠上面那个符号链接，JSON 里的绝对路径原样解析，**无需改任何数据文件**。

`python grpo/check_data.py --read` 全部 935 个文件用 open3d 实读通过，0 缺失 0 损坏。

### ④ 显存 ✅（其他进程已停）

8 张卡全空（每张 40441 MiB）。四卡实测峰值远低于上限，全量 AdamW 可跑。
若日后卡被占用，可用 `optim: adamw_bnb_8bit`（已在 `train_grpo.py` 加好透传）
或 `kl_coef: 0`（省掉整份 ref model）。

### ⑤ 学习信号 ✅（正式训练前最该看的一项）

新增 `grpo/probe_signal.py`，32 个真实 prompt × G=4 的采样结果：

| 指标 | 实测 | 说明 |
|---|---|---|
| `accuracy` | **0.859** | SFT 起点准确率 |
| `frac_nonzero_adv` | **0.344** | 能产生梯度的 group 占比，健康 |
| `parse_fail_rate` | 0.000 | 格式完全没崩 |
| `mean_completion_len` | 1.0 token | 模型直接吐单个字母 |

由于输出就是**单个字母**，附录里担心的"兜底正则 `\b([ABCD])\b` 误命中散文"
在本数据集上不成立。

### ⑥ 冒烟训练 ✅

```
单卡 3 步：step1 grad_norm=23.97，成功产出 ckpt
四卡 6 步：6/6 步 grad_norm 均非零（10.7~35.2），无 DDP 挂起
产出 ckpt 已验证可独立 from_pretrained 加载（1.830 B）
```

四卡约 **6.2 s/step**，有效 batch 16 prompt = 64 rollout。
13578 条 ÷ 16 ≈ 849 step/epoch → **单 epoch 约 90 分钟**。

单卡时会出现整步 `grad_norm=0.0`（4 个 micro-batch 全被 `std<=1e-6` 跳过），
属正常现象；四卡下 16 个 group 里几乎总有带信号的，未再出现。

---

## 八、建议的启动顺序

```bash
cd /home/aiscuser/nyp/3D-RL
source ~/miniconda3/etc/profile.d/conda.sh && conda activate spatiallm-grpo

# 1) 数据链路（不需要模型，最快）
python grpo/check_data.py            # 加 --read 会用 open3d 实读每个文件

# 2) 点云透传 + 采样 + 重算 log-prob（不需要数据集）
CUDA_VISIBLE_DEVICES=0 python grpo/smoke_rollout.py --num_generations 4 --max_new_tokens 64

# 3) 学习信号体检 —— 正式开训前最该看的一步
CUDA_VISIBLE_DEVICES=0 python grpo/probe_signal.py -n 32

# 4) 单卡 3 步冒烟
CUDA_VISIBLE_DEVICES=0 python grpo/train_grpo.py --config grpo/config_test.yaml --max_steps 3

# 5) 四卡正式跑
torchrun --nproc_per_node 4 grpo/train_grpo.py --config grpo/config_test.yaml
```

第 1、2 步是作者专门写来分段排障的，**别跳过**——它们能把"环境问题 / 数据问题 / 模型问题"三者隔离开。
第 3 步是本次新增的：GRPO 的梯度全部来自组内 reward 方差，`frac_nonzero_adv` 过低时
训练能跑完但一步都学不到东西，事前 2 分钟就能测出来。

---

## 附：当前实现的已知限制

调查过程中发现的、与"能否跑出效果"相关的设计限制，供调参时参考：

- **`num_iterations: 1` 是单步 on-policy**，ratio 恒为 1，`clip_eps` 永不触发。多步更新（复用 rollout 做多次梯度更新）尚未实现，调大该值无效。
- **reward 仅支持 A/B/C/D 选择题**（`reward.py` 全文只有 `[ABCD]` 正则）。grounding / bbox 任务喂进来会全部拿 0 分 → 整组被 `std<=1e-6` 跳过 → 一步梯度都不产生。grounding 目前只有 SFT（`configs/spatiallm_grounding.yaml`）。
- **`tok_total` 未跨卡同步**（`grpo_trainer.py:342`）：各卡回答长短不一导致分母不同，DDP 梯度平均后等价于给不同卡的 token 赋了不同权重。
- **整批被跳过时返回与模型无关的 0 loss**（`grpo_trainer.py:336-338`），该卡无参数进入计算图，配合 `ddp_find_unused_parameters=True` 在多卡下有挂起风险。
- **采样效率**：每个 group 做 1 次 generate + G 次 policy forward + G 次 ref forward，全部 batch=1 逐条跑。G=4 时是 9 次前向，可通过 padding 成 batch 提速约 4 倍。
- **advantage 除以 std** 会引入难度偏置（简单题 std 小 → advantage 被放大），Dr.GRPO 建议只减均值。
- **reward 兜底正则 `\b([ABCD])\b`** 理论上可能误命中，例如 "A UAV should collaborate" 里的 "A"。
  已用 `probe_signal.py` 实测：本 ckpt 输出就是**单个字母**（平均 1.0 token），
  `parse_fail_rate=0`，没有散文可供误命中，**本数据集上不成立**。换数据集或改 prompt
  导致模型开始输出完整句子时，需重新体检。

---

## 附二：本次为适配本机所做的代码改动

| 文件 | 改动 |
|---|---|
| `config_test.yaml` | `model_path` / `output_dir` 改为本机路径；修正"全量 2800 条"→ 13578 |
| `train_grpo.py` | docstring 路径；新增 `optim` 透传（默认不变，可选 `adamw_bnb_8bit`） |
| `smoke_rollout.py` | 默认路径；**删除本地重复的 `load_spatiallm`，改用 `spatiallm_grpo_utils` 的共享版本**——本地那份只支持 `dtype=`，在 transformers 4.53 上报 `__init__() got an unexpected keyword argument 'dtype'` |
| `grpo_trainer.py` | **把 tokenizer 转发给 HF Trainer 的 `processing_class`**——此前 tokenizer 被吞进 `self._tok`，导致存出的 ckpt 只有 `model.safetensors` 而没有 `tokenizer.json` / `vocab.json` / `merges.txt`，无法直接 `from_pretrained` |
| `check_data.py` | 新增本地模式（`--mode auto/local/blob`）与 `--read` 实读校验；原版只能查 blob |
| `probe_signal.py` | 新增：开训前的学习信号体检 |
