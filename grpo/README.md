# SpatialLM GRPO —— 运行可行性调查报告

> 调查日期：2026-08-15，环境就绪与实跑验证：2026-08-16
> 运行环境：本机 `/home/aiscuser/nyp/3D-RL`，8× A100-SXM4-40GB
> 结论：**已在本机跑通**。AirCop 8 卡（≈4.0 s/it）与 UrbanVideo 单卡均已实训，产出 ckpt 可独立加载。详见[第七节](#七本机就绪状态2026-08-16-实测)。

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

启动命令（完整版见[第八节](#八怎么手动跑起来复制即用)）：

```bash
# 多卡（AirCop 主力配置）
torchrun --nproc_per_node 8 grpo/train_grpo.py --config grpo/config_aircop.yaml

# 单卡调试（只跑 3 步）
CUDA_VISIBLE_DEVICES=0 python grpo/train_grpo.py --config grpo/config_test.yaml --max_steps 3
```

> ⚠️ 必须**在仓库根目录 `3D-RL/` 下运行**，因为 config 里的 `train_json: data/grpo_test.json` 是相对路径。

---

## 二、参数从哪里传入（三层）

### 第 1 层：YAML —— 唯一需要改的地方

YAML 是**全部超参的单一来源**。现有三个：`config_aircop.yaml`（8 卡主力）、
`config_urbanvideo.yaml`（单卡）、`config_test.yaml`（小样本调试）。

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

## 三、训练参数逐条讲解

> 这一节是写给**第一次做后训练**的人的。先说一句最重要的：
> GRPO 和 SFT 的直觉完全不同——SFT 是"照着标准答案抄"，loss 降就是在学；
> GRPO 是"自己答 G 遍，答对的那几遍加大概率、答错的减小概率"，
> **loss 恒等于 0 也完全正常**，真正要盯的是 `grpo/accuracy`。

现在有三个配置文件：`config_aircop.yaml`（主力）、`config_urbanvideo.yaml`、
`config_test.yaml`（小样本调试）。参数分 5 组，逐条说。

### ① 模型

```yaml
model_path: /home/aiscuser/nyp/ckpts/point_mixed_downsample   # SFT 产出的 ckpt，RL 在它基础上继续训
dtype: bfloat16                                                # 语言塔精度；点云塔强制 fp32（与推理一致）
```

RL 必须从一个**已经会做这个任务**的 SFT ckpt 出发。从随机权重开始，G 次采样全是
胡话、reward 全 0、组内没方差 → 一步梯度都产生不了。

### ② 数据

```yaml
train_json: data/grpo_aircop.json   # 相对仓库根目录
max_samples: null                   # null=全量；调试填 64，比 --max_steps 更省启动时间
num_bins: 1280                      # 点云离散化网格数，**必须与 SFT 时一致**，改了等于换了输入分布
max_points: 0                       # 点数上限，0=不限。见下方说明
```

`max_points` 是本轮新增的。体素下采样后的点数由场景尺度决定，各数据集差异极大：

| 数据集 | ply 文件均值 | 下采样后点数 |
|---|---|---|
| AirCop | 5.5 MB | p50 ≈ 2.0k，最大 < 6k |
| UrbanVideo | 105 MB | p50 ≈ 26k，p90 ≈ 59k，尾部见过 **22 万** |

点数直接决定点云编码器（Sonata，fp32）的显存。所以 AirCop 设 0（离红线很远），
UrbanVideo 设 65536（只影响约 6% 的文件，对它们做均匀随机下采样）。
注意点数**不等于** token 数——5.5 万点编码完只有约 156 个 token 进语言塔。

### ③ GRPO 采样

```yaml
num_generations: 8    # G，组大小。GRPO 最核心的参数
max_new_tokens: 8     # 答案就是 1 个字母，给 8 是留余量
temperature: 1.0      # 纯采样，不动分布
top_p: 1.0
top_k: 0
```

**`num_generations` (G) 是最该理解的一个。** GRPO 对同一道题采样 G 遍，
用这 G 个 reward 的均值当基线：`A_i = (r_i - mean) / std`。所以：

- G 太小 → 组内很容易 G 遍全对或全错 → `std = 0` → **整组被跳过，一点梯度都没有**
- G 太大 → 显存和时间线性增长

代码里 `std <= 1e-6` 就 `continue`，这就是为什么 `grpo/frac_nonzero_adv`
这个指标如此重要——它就是"没被跳过的组占比"。

`temperature: 1.0 / top_p: 1.0 / top_k: 0` 三个别动。截断分布会让采样不再是从
策略 π_θ 里真正抽样，policy gradient 就有偏了。这是 RL 和推理调参的关键区别：
**推理时你想要好答案所以截断，RL 采样时你要的是无偏样本。**

### ④ GRPO 目标

```yaml
kl_coef: 0.04       # β，KL 惩罚强度
clip_eps: 0.2       # PPO 的 clip 范围
num_iterations: 1   # 每批 rollout 更新几次
```

- **`kl_coef`（β）**：拴住策略别跑太远的绳子。训练时会**再加载一份冻结的原始模型**
  （ref model，约 3.6 GB 显存），每步算策略和它的 KL 散度，加进 loss 当惩罚。
  调大 = 更保守、更不容易崩，但学得慢。设 0 可以省掉那 3.6 GB，但没了缰绳，
  模型可能为了骗 reward 输出退化文本（reward hacking）。**新手保持 0.04。**
- **`clip_eps`**：`num_iterations=1` 时**完全不生效**（下面解释），先别管。
- **`num_iterations`**：采样一批 rollout 后，拿它做几次梯度更新。
  `=1` 就是纯 on-policy：采样用的策略 = 更新的策略 → 重要性比 ratio 恒等于 1
  → clip 永远不触发 → **loss 恒 ≈ 0**。这不是 bug，是数学上的必然（见第五节警告）。
  本实现只支持 1，调大无效。

### ⑤ 训练

```yaml
per_device_train_batch_size: 1   # 每卡每次 1 个 prompt = 1 个 group
gradient_accumulation_steps: 4
learning_rate: 1.0e-6            # 关键
num_train_epochs: 1
warmup_ratio: 0.03
lr_scheduler_type: constant
gradient_checkpointing: false    # 本机 40G A100 显存够，关掉换速度
dataloader_num_workers: 4
logp_batch_size: 4               # 一次前向算几条序列的 log-prob，0=整组一次算完
pcd_cache_size: 64               # 点云张量 LRU 缓存条数
seed: 42
```

**有效 batch 怎么算**：`卡数 × per_device_train_batch_size × gradient_accumulation_steps`。
8 卡时 = `8 × 1 × 4` = **32 个 prompt/step**，每个 prompt 采 G=8 遍
= **256 条 rollout/step**。AirCop 13578 条 ÷ 32 ≈ **424 step/epoch**。

**`learning_rate` 是最该小心的一个。** RL 阶段必须比 SFT 小**一到两个量级**
（SFT 用 2e-5，这里用 1e-6）。原因：SFT 有标准答案兜底，学偏了也偏不到哪去；
RL 的 reward 信号又稀疏又嘈杂（一个 0/1 标量要指导整个序列），
lr 一大策略立刻崩，而且**崩了不可逆**——采样分布坏了以后再也采不到好样本。
症状就是 `grpo/kl` 飙升。调参优先级：`learning_rate` > `num_generations` > `kl_coef`。

`logp_batch_size` 和 `pcd_cache_size` 是纯工程参数，不影响训练结果，只影响显存/速度。
`warmup_ratio: 0.03` + `constant` 表示前 3% 步线性升到 1e-6 然后一直保持。

### ⑥ 输出与日志

```yaml
output_dir: /home/aiscuser/nyp/saves/grpo_aircop
logging_steps: 1
save_steps: 200          # 每 200 步存一个 ckpt
save_total_limit: 3      # 最多留 3 个，自动删旧的
report_to: wandb
wandb_project: spatiallm-grpo
run_name: aircop-g8-lr1e6
```


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
saves/grpo_aircop/
├── checkpoint-200/          # save_steps=200，424 步/epoch 会存 2 个（save_total_limit=3 自动删旧）
├── config.json
├── model.safetensors        # 训完 save_model() 存的全量权重（非 LoRA，约 3.6GB）
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
 'grpo/frac_nonzero_adv': 1.0, 'grpo/skipped_groups': 1.0, 'grpo/mem_gb': 21.68}
```

### 各字段该期待什么值

| 字段 | 期望 | 说明 |
|---|---|---|
| `grpo/accuracy` | 起点 = SFT 在这批题上的准确率（AirCop ≈ 0.86，UrbanVideo ≈ 0.80）；**应缓慢上升** | **唯一真正的效果指标** |
| `grpo/frac_nonzero_adv` | 0.3~0.7 | 有多少组产生了学习信号。若 <0.1 说明题目太简单或太难，GRPO 学不动 |
| `grpo/kl` | 从 0 缓慢上升 | 若飙升（>0.5）说明策略跑偏，该调小 lr 或调大 kl_coef |
| `grpo/completion_len` | 1~2 | 选择题答案就 1 个字母。若接近 `max_new_tokens` 说明模型在胡扯没吐 EOS |
| `grpo/reward_std` | 0/1 reward 下的离散取值 | 全对/全错时为 0，该组会被跳过 |
| `grpo/skipped_groups` | 本步被跳过的组数 | 与 `frac_nonzero_adv` 互为补充 |
| `grpo/mem_gb` | 稳定在一个平台值 | 前几步会从 10 GB 爬到 21.7 GB（梯度和优化器状态逐步分配），**之后应该持平**。持续上涨才是异常 |
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

| 数据集 | blob 路径（相对 `output/liyan`） | 本地 |
|---|---|---|
| AirCop | `Pointcloud-VQA/AirCopBench/{Real2,Sim3,Sim5,Sim6}_VQA_train` | `/home/aiscuser/nyp/pcdata/Pointcloud-VQA/AirCopBench/...` |
| UrbanVideo | `Pointcloud-VQA/UrbanVideoBench/train_64` | `/home/aiscuser/nyp/pcdata/Pointcloud-VQA/UrbanVideoBench/train_64` |

符号链接 `/Pointcloud-VQA -> /home/aiscuser/nyp/pcdata/Pointcloud-VQA`，
JSON 里的绝对路径原样解析，**无需改任何数据文件**。

AirCop 935 个 `.ply`（5.41 GiB），`python grpo/check_data.py --read` 用 open3d 实读全部通过，0 缺失 0 损坏。
UrbanVideo 1159 个 `.ply`（118.5 GiB）。磁盘吃紧时可改用第九节的边下边训。

### ④ 显存 ✅

8 张卡各 40441 MiB。AirCop 8 卡实测平台 **21.7 GB**，余量充足。
UrbanVideo 单卡峰值 **28.3 GB**，8 卡会 OOM（详见"附：已知限制"）。

### ⑤ 学习信号 ✅（正式训练前最该看的一项）

`grpo/probe_signal.py` 在两个数据集上的实测：

| 指标 | AirCop (n=32, G=4) | UrbanVideo (n=48, G=8) |
|---|---|---|
| `accuracy` | **0.859** | **0.802** |
| `frac_nonzero_adv` | **0.344** | **0.333** |
| `parse_fail_rate` | 0.000 | 0.000 |
| `mean_completion_len` | 1.0 token | 1.0 token |

两者结论都是"学习信号充足，可以开训"。UrbanVideo 的 GT 覆盖 A/B/C/D/E/G，
模型预测覆盖 A–G，`reward.py` 放宽到 A–H 后解析零失败。

由于输出就是**单个字母**，附录里担心的"兜底正则误命中散文"在这两个数据集上都不成立。

### ⑥ 冒烟训练 ✅

```
单卡 3 步：step1 grad_norm=23.97，成功产出 ckpt
八卡 14 步（AirCop）：grad_norm 全程非零，accuracy 0.875 / 0.844 / 0.813，显存平台 21.68 GB
单卡（UrbanVideo）：accuracy 0.75 / 0.906，峰值 28.33 GB
产出 ckpt 已验证可独立 from_pretrained 加载（1.830 B）
```

AirCop 8 卡 **≈ 4.0 s/it**（点云编码共享修复前是 9.0 s/it），有效 batch 32 prompt = 256 rollout。
13578 条 ÷ 32 ≈ 424 step/epoch → **单 epoch 约 30 分钟**。

单卡时会出现整步 `grad_norm=0.0`（4 个 micro-batch 全被 `std<=1e-6` 跳过），
属正常现象；多卡下 32 个 group 里几乎总有带信号的，未再出现。

---

## 八、怎么手动跑起来（复制即用）

所有命令都**先进仓库根目录**。`grpo/train_grpo.py` 是相对路径，
在 `grpo/` 目录里执行会变成 `grpo/grpo/train_grpo.py` 而报 "can't open file"。

```bash
cd /home/aiscuser/nyp/3D-RL
```

另外：**别用系统的 `torchrun`**（会 `ModuleNotFoundError: No module named 'transformers'`），
要么先 `conda activate spatiallm-grpo`，要么直接写绝对路径
`/home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/torchrun`。下面统一用绝对路径，最不容易出错。

### 0) 一次性准备：wandb 登录

```bash
/home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/wandb login
```

粘贴 https://wandb.ai/authorize 的 token，存进 `~/.netrc`，以后不用再登。
机器连不上外网就在 config 里加 `wandb_offline: true`，事后 `wandb sync` 补传。

### 1) 开训前的两分钟体检（强烈建议）

```bash
CUDA_VISIBLE_DEVICES=0 /home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/python \
  grpo/probe_signal.py --config grpo/config_aircop.yaml -n 32
```

看 `frac_nonzero_adv`：低于 0.1 就别开训了，跑一整天也学不到东西。
（UrbanVideo 实测 n=48 时 accuracy 0.802 / frac_nonzero_adv 0.333，健康。）

### 2) AirCop —— 8 卡正式训练（**主力配置，已验证**）

```bash
cd /home/aiscuser/nyp/3D-RL && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/torchrun --nproc_per_node 8 \
  grpo/train_grpo.py --config grpo/config_aircop.yaml
```

实测 **≈ 4.0 s/it**，显存平台 21.7 GB / 40 GB，有效 batch 32 prompt = 256 rollout，
13578 条 ÷ 32 ≈ **424 step/epoch**，单 epoch 约 30 分钟。

### 3) UrbanVideo —— **单卡**（8 卡会 OOM，见第九节）

```bash
cd /home/aiscuser/nyp/3D-RL && \
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/python \
  grpo/train_grpo.py --config grpo/config_urbanvideo.yaml
```

实测峰值 28.33 GB / 40 GB。

### 4) 挂后台跑 + 看日志

两种都行，推荐 tmux（可以随时回去看实时输出、Ctrl-C 也方便）。

**tmux：**

```bash
tmux new -s grpo                      # 建会话
# —— 在会话里粘上面第 2 步的命令 ——
# Ctrl-B 然后按 D 脱离，训练继续跑
tmux attach -t grpo                   # 随时回来看
tmux ls                               # 看有哪些会话
```

**nohup（不想装/学 tmux 时）：**

```bash
mkdir -p ~/nyp/logs
cd /home/aiscuser/nyp/3D-RL && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/torchrun --nproc_per_node 8 \
  grpo/train_grpo.py --config grpo/config_aircop.yaml \
  > ~/nyp/logs/aircop.log 2>&1 &
echo $!                               # 记下 PID，要停就 kill 它
```

监控：

```bash
tail -f ~/nyp/logs/aircop.log         # 实时日志
grep grpo/accuracy ~/nyp/logs/aircop.log | tail -20   # 只看准确率
nvitop                                # 看显存/利用率
```

wandb 网页端会实时出曲线（项目 `spatiallm-grpo`，run 名就是 config 里的 `run_name`）。
**盯 `grpo/accuracy`，不要盯 loss**——理由见第五、六节。

要停止：

```bash
pkill -f train_grpo.py                # torchrun 会拉起 8 个子进程，按脚本名杀干净
```

### 5) 断点续训

`save_steps` 到点会在 `output_dir` 下写 `checkpoint-<step>/`（只留最近 3 个）。
崩了或手动停了之后，在原命令后加一个参数即可从断点继续，
优化器状态、lr schedule、数据顺序都会恢复：

```bash
cd /home/aiscuser/nyp/3D-RL && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/torchrun --nproc_per_node 8 \
  grpo/train_grpo.py --config grpo/config_aircop.yaml \
  --resume_from_checkpoint /home/aiscuser/nyp/saves/grpo_aircop/checkpoint-200
```

### 6) 分段排障（跑不起来时按顺序试）

```bash
python grpo/check_data.py                        # 只查数据链路，不加载模型；--read 用 open3d 实读
CUDA_VISIBLE_DEVICES=0 python grpo/smoke_rollout.py --num_generations 4   # 只测采样，不要数据集
CUDA_VISIBLE_DEVICES=0 python grpo/train_grpo.py --config grpo/config_test.yaml --max_steps 3
```

这三步专门用来把"环境问题 / 数据问题 / 模型问题"隔离开，别跳过。


---

## 九、边下边训（stream_from_blob）

点云数据总量很大（UrbanVideo 一个数据集就 118.5 GB），全量拉到本地既慢又占盘。
`grpo/blob_stream.py` 实现了 **下一批 → 训一批 → 删一批** 的流式采样器。
数据已经落盘时保持 `stream_from_blob: false` 即可，两条路径可随时切换。

### 怎么开

```yaml
stream_from_blob: true
pcd_local_root: /home/aiscuser/nyp/pcdata
stream_window_files: 24     # 每个窗口下载多少个 ply
stream_prefetch: true       # 后台线程预取下一窗口，把下载时间藏进训练时间
stream_delete_after: true   # 窗口训完就删（只删本进程下载的文件）
stream_release_lag: 2       # 滞后 2 个窗口再删，防止预取/训练错位时误删
```

### 它是怎么工作的

按点云文件分组打乱（webdataset 风格的两级 shuffle）：
文件级先 shuffle → 取 `window_files` 个文件构成一个窗口 → 窗口内的样本再 shuffle。
这样既保证随机性，又保证同一时刻只有少数几个 ply 需要在盘上。
磁盘峰值 ≈ `(release_lag + 2) × window_files × 单文件大小`。

多卡时窗口按 rank 切分，每个 rank 只下载自己那份，不重复。

### 为什么每个数据集的 window 不一样

`stream_window_files` 不能取一个通用值——各数据集单文件大小差 30 倍。实测：

| 数据集 | 文件数 | 单文件均值 | 合计 | `stream_window_files` | 单窗口 | 磁盘峰值 |
|---|---|---|---|---|---|---|
| AirCop | 935 | 5.5 MB | 5.4 GB | **192** | ≈ 1.1 GB | ≈ 4 GB |
| UrbanVideo | 1159 | 104.7 MB（p50 124.8） | 118.5 GB | **24** | ≈ 2.5 GB | ≈ 10 GB |

原则：**让单窗口落在 1~3 GB**。窗口太小 → shuffle 退化成近似顺序读，
同一窗口内样本高度相关，梯度有偏；窗口太大 → 磁盘占用逼近全量，等于没做流式。
换新数据集时先量一下文件大小分布再定这个数，别照抄。

### 注意

- 下载凭据从 `~/.blob_config.json` 读（SAS token），脚本刻意不把 URL 打进错误信息，避免 token 进日志。
- `stream_delete_after` 只删本进程下载的文件；手动预先放好的数据不会被误删。
- 预取线程失败时会退化成同步下载并打警告，不会静默跳过样本。

---

## 十、训完怎么测 checkpoint

### 先说清楚仓库里**没有**什么

- **根目录的 `eval.py` 用不了。** 它是 SpatialLM 原版的**室内布局估计**评测——算
  wall/door/window 和 20 类家具 bbox 的 F1@IoU0.25/0.50，输入是 layout txt 文件对。
  和选择题准确率毫无关系。`inference.py`（`generate_layout`）同理。这两个是上游
  SpatialLM 留下的，跟 GRPO 这条线没有交集。
- **没有 val/test split。** `data/AirCopBench/` 下四个全是 `*_VQA_train.json`，
  UrbanVideo 只有一个 MCQ json。且训练配置是 `max_samples: null`，**全量参与训练**。

所以下面这个脚本抽的是**训练集的子集**：accuracy 涨了只能说明"没训坏 / 在见过的题上更准了"，
**不能证明泛化**。要证明泛化必须先切一份留出集重训——目前选择是不切。

### `grpo/eval_mcq.py`

```bash
# 单个 ckpt
CUDA_VISIBLE_DEVICES=0 /home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/python \
  grpo/eval_mcq.py --config grpo/config_aircop.yaml -n 512

# 多个 ckpt 对比（跑在完全相同的题上，最后出对比表）
CUDA_VISIBLE_DEVICES=0 /home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/python \
  grpo/eval_mcq.py --config grpo/config_aircop.yaml -n 2000 \
    --ckpt /home/aiscuser/nyp/ckpts/point_mixed_downsample \
    --ckpt /home/aiscuser/nyp/saves/grpo_aircop/checkpoint-200 \
    --ckpt /home/aiscuser/nyp/saves/grpo_aircop
```

**第一个 `--ckpt` 传 SFT 原始权重当基线**，对比表里的 Δ 就是 GRPO 带来的增量。

与 `probe_signal.py` 的分工：probe 开训**前**用（随机采样 G 条，看
`frac_nonzero_adv` 够不够学）；eval_mcq 训完**后**用（贪心解码，看 accuracy 涨没涨）。
关键差别是 **`do_sample=False`**——probe 走随机采样，同一 ckpt 每次跑结果都不同，
不能用来做 ckpt 间比较。这里必须贪心。所有 ckpt 用同一个 `--seed` 抽同一批题。

### 输出长什么样（AirCop n=60 实测）

```
  overall accuracy     = 0.9333  (56/60)
  parse_fail_rate      = 0.0000
  mean_completion_len  = 1.0 tokens
  众数基线（全答 B）  = 0.4000   <- 低于这个说明模型没在做题

  按数据集 split:
    Sim3_VQA_train       0.9459   (35/37)
    Sim6_VQA_train       1.0000   (10/10)
    Real2_VQA_train      0.7778   (7/9)
    ...
  按 GT 字母（该字母的题答对了多少）:
    A  0.9545 (21/22)   B  0.9167 (22/24)   C  1.0000 (9/9)   D  0.8000 (4/5)

  GT   分布 = {'A': 22, 'B': 24, 'C': 9, 'D': 5}
  预测 分布 = {'A': 23, 'B': 23, 'C': 10, 'D': 4}
```

三个分项各自回答一个问题：

- **众数基线** —— AirCop 的 GT 分布很偏（A 占 44.7%），无脑全答 A 就有 0.447。
  accuracy 掉到这个数附近 = 策略塌成了常数，不是"效果一般"而是训崩了。
- **按 split** —— AirCop 的 Real2 是真实数据、Sim* 是仿真，分开看能发现"仿真涨了但真实掉了"这种情况。
  UrbanVideo 自动按文件名前缀拆成 EmbodiedCity / AerialVLN。
- **GT 分布 vs 预测分布** —— 两行越接近越好。预测明显塌向某个字母是训崩的典型征兆，
  而且它比 accuracy 下降出现得更早。

贪心解码下 AirCop ≈ **0.24 s/条**：n=512 约 2 分钟，n=2000 约 8 分钟。
UrbanVideo 慢得多（单个 ply 125 MB，读盘是瓶颈）。
n 越小噪声越大，想区分 1~2 个百分点的差距建议 `-n 2000` 以上。

> 注：贪心 accuracy 会明显高于 `probe_signal.py` 的采样 accuracy
>（AirCop 实测 0.933 vs 0.859），这是解码方式的差异，不是模型变好了。**只和贪心比贪心。**



调查过程中发现的、与"能否跑出效果"相关的设计限制，供调参时参考：

- **UrbanVideo 上不了多卡**（本轮实测）：单卡峰值 28.3 GB 可跑，8 卡稳定 OOM 在约 38 GB。
  静态占用（权重 + ref + 梯度 + 优化器状态）爬坡到 21.7 GB 后，点云编码器的激活尖峰
  再叠 DDP 梯度桶就顶穿 40 GB。**已试过且无效**：`max_points` 65536→16384、
  `num_generations` 8→4、`adamw_bnb_8bit`、`gradient_checkpointing`、`expandable_segments`
  ——峰值都稳在 38 GB。真要上多卡得砍掉编码器激活本身（冻结点云塔，或让它走 bf16），
  这会改动 SFT 共用的模型代码，本轮没做。
- **`num_iterations: 1` 是单步 on-policy**，ratio 恒为 1，`clip_eps` 永不触发。多步更新（复用 rollout 做多次梯度更新）尚未实现，调大该值无效。
- **reward 只支持单选题**（`reward.py` 的 `CHOICES` 现已放宽到 A–H，覆盖 AirCop 的 A–D 和 UrbanVideo 的 A–G）。grounding / bbox 任务喂进来会全部拿 0 分 → 整组被 `std<=1e-6` 跳过 → 一步梯度都不产生。grounding 目前只有 SFT（`configs/spatiallm_grounding.yaml`）。
- **`tok_total` 未跨卡同步**：各卡回答长短不一导致分母不同，DDP 梯度平均后等价于给不同卡的 token 赋了不同权重。
- **整批被跳过时返回与模型无关的 0 loss**，该卡无参数进入计算图，配合 `ddp_find_unused_parameters=True` 在多卡下有挂起风险。
  （`ddp_find_unused_parameters=False` 不能开：零方差组会跳过整个 micro-batch，确实会留下没有梯度的参数。）
- **advantage 除以 std** 会引入难度偏置（简单题 std 小 → advantage 被放大），Dr.GRPO 建议只减均值。
- **reward 兜底正则** 理论上可能误命中，例如 "A UAV should collaborate" 里的 "A"。
  已用 `probe_signal.py` 实测：本 ckpt 输出就是**单个字母**（平均 1.0 token），
  `parse_fail_rate=0`，没有散文可供误命中，**两个数据集上都不成立**。换数据集或改 prompt
  导致模型开始输出完整句子时，需重新体检。

---

## 附二：本次为适配本机所做的代码改动

### 第一轮：跑通

| 文件 | 改动 |
|---|---|
| `config_test.yaml` | `model_path` / `output_dir` 改为本机路径；修正"全量 2800 条"→ 13578 |
| `train_grpo.py` | docstring 路径；新增 `optim` 透传（默认不变，可选 `adamw_bnb_8bit`） |
| `smoke_rollout.py` | 默认路径；**删除本地重复的 `load_spatiallm`，改用 `spatiallm_grpo_utils` 的共享版本**——本地那份只支持 `dtype=`，在 transformers 4.53 上报 `__init__() got an unexpected keyword argument 'dtype'` |
| `grpo_trainer.py` | **把 tokenizer 转发给 HF Trainer 的 `processing_class`**——此前 tokenizer 被吞进 `self._tok`，导致存出的 ckpt 只有 `model.safetensors` 而没有 `tokenizer.json` / `vocab.json` / `merges.txt`，无法直接 `from_pretrained` |
| `check_data.py` | 新增本地模式（`--mode auto/local/blob`）与 `--read` 实读校验；原版只能查 blob |
| `probe_signal.py` | 新增：开训前的学习信号体检 |

### 第二轮：wandb / 提速 / 两个新数据集 / 边下边训

| 文件 | 改动 |
|---|---|
| `grpo_trainer.py` | **`_share_point_encoding`**：`spatiallm_qwen3.forward` 对 batch 里每个元素单独调 `forward_point_cloud`，而一个 group 的 G 条序列共用同一片点云 → 同一朵云被重复编码 G 次。加了个 memo 上下文管理器让组内共享一次编码结果。**这是本轮最重要的改动**：AirCop 从 9.0 s/it 降到 **4.0 s/it**，UrbanVideo 从 OOM 变成单卡可跑（5.5 万点：generate 5.00 GB / 前向+反向 14.34 GB） |
| `grpo_trainer.py` | ref model 改为在 `__init__` 里就搬上卡。原来惰性搬迁的时机恰好是策略前向图还挂着的显存最高点，那时再要 3.6 GB 常常正好 OOM |
| `grpo_trainer.py` | 新增 `max_points` 参数；`_flush_logs` 新增 `grpo/mem_gb` 指标 |
| `grpo_trainer.py` | `_logprobs_chunked` 把共享上下文开在分块循环**外面**，否则分块之间又重复编码 |
| `spatiallm_grpo_utils.py` | `load_point_cloud_tensor` 新增 `max_points`：超限时做均匀随机下采样，且**保序**（不打乱 z-order 序列化的局部性） |
| `reward.py` | `CHOICES` 从 `[ABCD]` 放宽到 `[A-H]`，支持 UrbanVideo 的 A–G |
| `config_aircop.yaml` | 新增：AirCop 8 卡主力配置（`max_points: 0`，`stream_window_files: 192`） |
| `config_urbanvideo.yaml` | 新增：UrbanVideo 单卡配置（`max_points: 65536`，`stream_window_files: 24`），header 记录了 OOM 现状与所有试过无效的办法 |
| `blob_stream.py` | 新增：边下边训采样器（文件级 shuffle + 窗口内 shuffle + 预取 + 滞后删除） |
| `eval_mcq.py` | 新增：选择题 ckpt 评测（贪心解码、多 ckpt 对比、按 split / 按 GT 字母拆解 + 众数基线）。根目录 `eval.py` 是布局估计 F1，与选择题无关，用不了 |
| `train_grpo.py` | 透传 `max_points` / `stream_sampler`；wandb 环境变量提前注入；训练结束打印峰值显存 |
| `.gitignore` | 追加 `wandb/` |

