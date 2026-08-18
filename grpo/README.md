# SpatialLM GRPO —— 运行可行性调查报告

> 📊 **实验记录与分析在单独的 [EXPERIMENTS.md](EXPERIMENTS.md)**，本文档只讲怎么配、怎么跑、怎么排障。
> 新的实验发现请追加到 EXPERIMENTS.md，格式是固定的五段：现象 / 假设 / 实验 / 结果 / 结论与动作。
>
> ⚠ 2026-08-18 起必读 [EXPERIMENTS.md 的结论汇总](EXPERIMENTS.md#当前结论汇总)：
> 实测两个数据集的题**几乎不依赖点云**（87.5 分里点云只值 1.5 分），
> 在它们上面调超参和奖励函数没有意义。动手改训练配置之前先读那一节。
>
> 调查日期：2026-08-15，环境就绪与实跑验证：2026-08-16，显存问题定位与修复：2026-08-17
> 运行环境：本机 `/home/aiscuser/nyp/3D-RL`，8× A100-SXM4-40GB
> 结论：**已在本机跑通**。AirCop 与 UrbanVideo **均可 8 卡训练**，产出 ckpt 可独立加载。详见[第七节](#七本机就绪状态2026-08-16-实测)。
>
> ⚠ 2026-08-17 更新：此前"UrbanVideo 只能单卡 / 8 卡必 OOM"的结论**是错的**，
> 而且"单卡可跑"同样是错的（实测第 7~9 步 OOM）。错误的根因、当时是怎么误判的、
> 以及最终的解法，全部写在[附三：显存排查的四个坑](#附三显存排查的四个坑2026-08-17)。
> 如果你之后再遇到 OOM，**先读附三**，能省掉我这次走的所有弯路。


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
9. [边下边训](#九边下边训stream_from_blob)
10. [训完怎么测 checkpoint](#十训完怎么测-checkpoint)
11. [难度打标](#十一难度打标difficulty_log)
12. [附一：已知限制](#附一已知限制)
13. [附二：代码改动](#附二本次为适配本机所做的代码改动)
14. [**附三：显存排查的四个坑**](#附三显存排查的四个坑2026-08-17) ← OOM 了先看这个

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
gradient_checkpointing: false    # AirCop 显存够，关掉换速度；UrbanVideo 必须 true
dataloader_num_workers: 4
logp_batch_size: 4               # 一次前向算几条序列的 log-prob，0=整组一次算完
pcd_cache_size: 64               # 点云张量 LRU 缓存条数
seed: 42
```

**有效 batch 怎么算**：`卡数 × per_device_train_batch_size × gradient_accumulation_steps`。
8 卡时 = `8 × 1 × 4` = **32 个 prompt/step**，每个 prompt 采 G=8 遍
= **256 条 rollout/step**。AirCop 13578 条 ÷ 32 ≈ **424 step/epoch**；
UrbanVideo 4080 条 ÷ 32 ≈ **128 step/epoch**（差一个数量级，`save_steps` 别照抄）。

**`learning_rate` 是最该小心的一个。** RL 阶段必须比 SFT 小**一到两个量级**
（SFT 用 2e-5，这里用 1e-6）。原因：SFT 有标准答案兜底，学偏了也偏不到哪去；
RL 的 reward 信号又稀疏又嘈杂（一个 0/1 标量要指导整个序列），
lr 一大策略立刻崩，而且**崩了不可逆**——采样分布坏了以后再也采不到好样本。
症状就是 `grpo/kl` 飙升。调参优先级：`learning_rate` > `num_generations` > `kl_coef`。

`logp_batch_size` 和 `pcd_cache_size` 是纯工程参数，不影响训练结果，只影响显存/速度。
`warmup_ratio: 0.03` + `constant` 表示前 3% 步线性升到 1e-6 然后一直保持。

**⚠ 显存相关的四个参数，UrbanVideo 上必须一起改，缺一个就 OOM**（见附三）：
`gradient_checkpointing: true`、`max_points: 16384`、`logp_batch_size: 1`、
`optim: adamw_bnb_8bit`。`gradient_checkpointing` 会影响速度但不影响结果；
`max_points` **会影响结果**（降低输入保真度），是为跑通做的取舍。


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

**⚠ `save_steps` 必须按"这个数据集在这个卡数下有多少步/epoch"来定，不能跨数据集照抄。**
8 卡时 AirCop 是 424 步/epoch（`save_steps: 200` → 存 200/400/424 三个），
UrbanVideo 只有 **128 步/epoch**（`save_steps: 200` → **一个中间 ckpt 都存不下**，
只有训练结束那一个，等于白跑，没法做早停对比）。UrbanVideo 已改成 `save_steps: 25`。
`save_total_limit` 保留的是**最后 N 个**，删的永远是最早的——而 AirCop 实测 accuracy
峰值出现在前 1/5，所以它不能设得比实际产出的 ckpt 数更小，否则最好的那个会被删掉。



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
 'grpo/frac_nonzero_adv': 1.0, 'grpo/skipped_groups': 1.0,
 'grpo/mem_gb': 24.94,     'grpo/mem_now_gb': 19.65}
```

### 各字段该期待什么值

| 字段 | 期望 | 说明 |
|---|---|---|
| `grpo/accuracy` | 起点 = SFT 在这批题上的准确率（AirCop ≈ 0.86，UrbanVideo ≈ 0.80）；**应缓慢上升** | **唯一真正的效果指标** |
| `grpo/frac_nonzero_adv` | 0.3~0.7 | 有多少组产生了学习信号。若 <0.1 说明题目太简单或太难，GRPO 学不动 |
| `grpo/kl` | 从 0 缓慢上升 | 若飙升（>0.5）说明策略跑偏，该调小 lr 或调大 kl_coef |
| `grpo/completion_len` | 1~2 | 选择题答案就 1 个字母。若接近 `max_new_tokens` 说明模型在胡扯没吐 EOS |
| `grpo/reward_std` | 0/1 reward 下的离散取值 | 全对**或全错**时为 0，该组会被跳过（两头都跳，不只是全对） |
| `grpo/skipped_groups` | 本步被跳过的组数 | 与 `frac_nonzero_adv` 互为补充 |
| `grpo/mem_gb` | **步内真峰值**，忽高忽低属正常，但顶不能逼近卡容量 | 用 `max_memory_allocated()` 且每步 `reset_peak_memory_stats()`。OOM 由**瞬时峰值**决定，波动来自零方差跳过（跳过的步不建反向图，显存就低）。**留余量要按"所有组都产生梯度"的最坏情况算**，不能指望跳过 |
| `grpo/mem_now_gb` | 稳定在一个平台值 | 步末常驻显存。前几步从 10 GB 爬到平台值（梯度和优化器状态逐步分配），之后应持平；**持续上涨才是泄漏**。它和 `mem_gb` 两条一起看才能区分"常驻涨"和"瞬时尖峰"——只看这一条会漏掉尖峰，我就是这么误判的（见附三） |
| `loss` | **≈ 0，且没有意义** | 见下方警告 |

> ⚠️ **别看 loss 曲线。** `num_iterations=1` 时 ratio ≡ 1，per-token loss = −A_g；而组内 advantage 之和恒为 0，所以 `loss_sum` 天然接近 0（只有各条长度不等时才有微小偏离）。**loss 不降不代表没在学**，GRPO 的 loss 是个 surrogate，请只盯 `grpo/accuracy` 和 `grpo/kl`。

---

## 六、监控方式（wandb）

`config_aircop.yaml` / `config_urbanvideo.yaml` 已默认 `report_to: wandb`。

```yaml
report_to: wandb
wandb_project: spatiallm-grpo-aircop   # 项目名（train_grpo.py 会写进 WANDB_PROJECT）
run_name: aircop-g8-lr1e6              # run 名，启动时会自动追加时间戳
# wandb_run_id: xxxx                   # 只在想续跑某个已有 run 时才填
# wandb_offline: true                  # 机器连不上外网时改这个，事后 wandb sync 上传
```

**一次训练 = 一条独立的 wandb 条目**，靠两层保证：

| 层级 | 取值 | 作用 |
|---|---|---|
| project（面板） | `spatiallm-grpo-aircop` / `spatiallm-grpo-urbanvideo` | 两个数据集各一个面板，**永远不会混**。两者的 accuracy 基线本来就不可比（4 选 1 vs 7 选 1），没有放一起的必要 |
| run（条目） | `urbanvideo-g8-lr1e6-0817-0812` | `train_grpo.py` 自动给 `run_name` 追加 `%m%d-%H%M`。同一数据集训多次也分得清是哪一次，且能在同一面板里叠着比 |

run id 则每次由 wandb 随机生成（**前提是 `WANDB_RUN_ID` 已被清掉**，见下方警告）。

首次使用要先登录一次（只需一次，token 存进 `~/.netrc`）：

```bash
/home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/wandb login
```

自定义的 `grpo/*` 键走 `self.log()`，会自动进 wandb，不需要额外改代码。
事后画曲线也可以直接读 `output_dir/trainer_state.json` 里的 `log_history`。

**⚠ 本机容器预置了一整套 `WANDB_*` 环境变量，会劫持所有 run**（2026-08-17 踩到）：

```
WANDB_RUN_ID=7213779716.62777-72df8069-dba0    ← 固定的 run id，才是罪魁祸首
WANDB_PROJECT=vllm-sh-wanli
WANDB_NAME=4x-palisades33-LLM2CLIP-yif-unirun-40Ga100
WANDB_RUN_GROUP / WANDB_NOTES
```

wandb 会自动读 `WANDB_RUN_ID`，于是**每次启动训练都挂到同一个 run 上**——本机
`wandb/` 下 15 个 run 目录全是同一个 id。症状是：打开页面看到的是历次启动叠在一起的
曲线，而新一轮的 step 从 1 重新开始、低于服务端已有的最大 step，图上看着就像
"还是上一轮的曲线，而且不实时刷新"。另外 `WANDB_PROJECT` 会让 run 跑到
`vllm-sh-wanli` 里去，`spatiallm-grpo` 项目下什么都看不到。

`train_grpo.py` 已在 `wandb.init` 之前主动清掉 `WANDB_RUN_ID` / `WANDB_RESUME` /
`WANDB_RUN_GROUP` / `WANDB_NOTES`，并把 `WANDB_PROJECT` 由 `setdefault` 改成强制赋值
（原来的 `setdefault` 拗不过环境变量）。想续跑某个 run 时在 config 里填
`wandb_run_id: xxxx` 即可。

已经被污染的本地 run 数据不会丢，训练结束后可以重新灌进一个干净的 run：

```bash
wandb sync wandb/run-<时间戳>-<旧id> --id <随便起个新id> -p spatiallm-grpo-urbanvideo
```

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

8 张卡各 40441 MiB。

| 配置 | 8 卡实测 | 余量 |
|---|---|---|
| AirCop（`config_aircop.yaml`） | 常驻平台 **21.7 GB** | 充足 |
| UrbanVideo（`config_urbanvideo.yaml`，2026-08-17 修复后） | 跑满 128 步实测峰值 **35.7 GB**（15 步冒烟只见到 34.4） | 约 3.8 GB |

UrbanVideo 那套是四项改动叠加的结果，缺一项就重新 OOM，详见[附三](#附三显存排查的四个坑2026-08-17)。

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
八卡 15 步（UrbanVideo，2026-08-17 修复后）：无 OOM，kl 0.0002~0.0046 正常，
  峰值序列 20.7 24.9 28.5 19.7 24.6 24.3 34.4 25.2 24.8 19.9 28.4 23.2 24.9 25.2 23.4
产出 ckpt 已验证可独立 from_pretrained 加载（1.830 B）
```

AirCop 8 卡 **≈ 4.0 s/it**（点云编码共享修复前是 9.0 s/it），有效 batch 32 prompt = 256 rollout。
13578 条 ÷ 32 ≈ 424 step/epoch → **单 epoch 约 30 分钟**。
UrbanVideo 4080 条 ÷ 32 ≈ **128 step/epoch**，单 epoch 约 20 分钟。

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

### 3) UrbanVideo —— 8 卡（2026-08-17 修复后可用）

```bash
cd /home/aiscuser/nyp/3D-RL && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/aiscuser/miniconda3/envs/spatiallm-grpo/bin/torchrun --nproc_per_node 8 \
  grpo/train_grpo.py --config grpo/config_urbanvideo.yaml
```

实测跑满 1 个 epoch 无 OOM，峰值 **35.7 GB / 39.5 GB**，余量约 3.8 GB。
4080 条 ÷ 32 ≈ **128 step/epoch**，单 epoch 实测 28.9 分钟。

⚠ 余量只有 3.8 GB，比 AirCop 紧得多。开跑后**盯一下 wandb 的 `train/grpo/mem_gb`**（真峰值），
逼近 39 就要停下来降 `max_points`。不要在同一批卡上并行跑别的任务。
（15 步冒烟测试当时只见到 34.4 GB，那个 35.7 的尖峰要跑满 128 步才出现——
又一次印证了[附三坑 4](#坑-4短测试给出假的能跑结论)。）

⚠ 这个配置里 `max_points: 16384` 会把一半以上样本下采样（本数据集 p50=26.2k），
输入保真度低于 SFT 时，是为跑通做的取舍。理由和备选方案见[附三](#附三显存排查的四个坑2026-08-17)。

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
- **仓库里原本没有 val/test split。** `data/AirCopBench/` 下四个全是 `*_VQA_train.json`，
  UrbanVideo 只有一个 MCQ json。`data/grpo_test.json` 名字骗人，里面是 AirCop 的**训练**数据。
  `data/nuScenes/*val*.json`、`data/DVG/test.json` 属于别的任务，跟这条线无关。

**2026-08-17 起可以切了**：`grpo/split_holdout.py` 按点云分组切留出集（seed 42 确定性）。

```bash
python grpo/split_holdout.py --json data/UrbanVideoBench/MCQ_EmbodiedCity_AerialVLN.json \
    --test_ratio 0.2 --seed 42
# -> _train.json 3251 条 / 927 点云   _test.json 829 条 / 232 点云   点云交集 0
```

**必须按点云分组，不能按条目随机切**：UrbanVideo 4080 题只有 1159 个点云（平均 3.5 题/场景，
max 11），按条目切会让同一场景同时出现在两边，泄漏。脚本还按来源分层
（AerialVLN / EmbodiedCity 准确率差 13 个点，比例飘了整体数字就跟着飘），并断言两边点云交集为 0。

⚠ 切完**必须用 train 那份重训**。对已用全量数据训完的 ckpt 跑这个 test，测的还是训练集。
配套：`config_urbanvideo_holdout.yaml`（训练，指向 `_train.json`）+
`config_urbanvideo_eval_test.yaml`（评测，指向 `_test.json`）。

⚠ 829 条的检测下限（McNemar，p<0.05 双侧）：不一致率 5% → 需净提升 ≥1.7 点；10% → ≥2.3 点；
20% → ≥3.2 点。真实提升低于这个幅度会显示"不显著"，**那是样本量不够，不等于没提升**。

若仍用不切分的全量配置（`config_urbanvideo.yaml` / `config_aircop.yaml`），下面脚本抽的就是
**训练集的子集**：accuracy 涨了只能说明"没训坏"，**不能证明泛化**。

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

### UrbanVideo 首轮 RL 的实测结果（2026-08-17，n=1000）

第一次完整跑完（1 epoch = 128 步，lr 1e-6）后的评测，**结论是 RL 什么也没改变**：

| ckpt | accuracy | vs base 的分歧题 | McNemar p |
|---|---|---|---|
| base（SFT） | 0.8800 | — | — |
| checkpoint-25 | 0.8800 | 2 题（b=1 c=1） | 1.00 |
| checkpoint-75 | 0.8790 | 3 题（b=2 c=1） | 1.00 |
| checkpoint-128 | 0.8800 | 4 题（b=2 c=2） | 1.00 |

1000 题里 996~999 题预测**逐字相同**，净差为 0。这不是"提升不显著"，是"基本没动"。
三条独立证据指向同一结论：

1. **权重挪动量太小**：`||ΔW||/||W||` 在 checkpoint 之间只有 **1e-5 量级**。
   （注意别被 "c25 vs base = 1.66e-3" 骗了——base 存的是 fp32，训练按 bf16 加载，
   光 fp32→bf16 取整就有约 1.6e-3 的相对误差，那一项全是精度噪声不是学习。）
2. **步数太少**：4080 条 ÷ 32 = **只有 128 步**。AirCop 同样配置有 424 步。
3. **一大半 group 被跳过**：base 单题正确率约 0.88，G=8 时全对的概率
   `0.88^8 ≈ 0.36`；EmbodiedCity split 已到 0.965，`0.965^8 ≈ 0.75` 直接被跳过。
   训练日志里 `frac_nonzero_adv` 常年在 0.25~0.75，即每步 4 个 group 有 1~3 个不产生梯度。

模型本身是正常的（`parse_fail_rate=0`，众数基线 0.22，预测分布与 GT 分布几乎重合，
没有塌向任何字母）。要让 RL 真正起作用，按杠杆大小排序：

- **提 `learning_rate`**（1e-6 → 3~5e-6，仍比 SFT 的 2e-5 低一个量级）
- **加 `num_train_epochs`**（1 → 4，即 512 步）
- **过滤掉太简单的题**：把 base 采样 8 次全对的样本剔出训练集，
  直接抬高 `frac_nonzero_adv`。EmbodiedCity 已接近饱和，贡献的梯度极少。



## 十一、难度打标（`difficulty_log`）

训练本来就要给每个 prompt 采 G 条 rollout 并逐条判对错，这份信息原先聚合成 accuracy
之后就扔了。打开开关后把**每个 group 的逐题结果**落盘，于是跑一遍训练顺带得到一份
全数据集难度图谱。开销：每 group 一行 json，纯 CPU 无同步，可忽略。

```yaml
difficulty_log: true      # 默认 false；不写这行就是关的
```

产物在 **`<output_dir>/difficulty/rank{0..7}.jsonl`**（8 个 rank 各写各的，不加锁——
并发追加同一文件会撕裂）。每行：

```json
{"idx": 142, "step": 37, "epoch": 1.16, "n_correct": 3, "n_gen": 8,
 "gt": "B", "preds": ["B","D","B","A","B","D","D","A"], "pcd": "..."}
```

落盘的是**原始计数**，不是标签——阈值随时可改，不用重训。记录点在
`grpo_trainer.py` 的 `if std <= 1e-6: continue` **之前**：被跳过的组正是"全对"和"全错"，
写在 continue 之后就全丢了。存 `preds` 是为了区分**错成同一个选项**（系统性错误信念）
和**散着错**（纯不会），两者要的干预完全不同。

⚠ 文件以 `"a"` 追加模式打开。同一个 `output_dir` 跑第二次，两轮记录会混在一起——
重跑前换 `output_dir` 或先把 `difficulty/` 挪走。

### `grpo/analyze_difficulty.py`

```bash
python grpo/analyze_difficulty.py --dir <output_dir>/difficulty --max_epoch 1.0 --show 10

# 导出难题子集
python grpo/analyze_difficulty.py --dir ... --max_epoch 1.0 \
    --src data/UrbanVideoBench/MCQ_EmbodiedCity_AerialVLN_train.json \
    --emit_hard data/UrbanVideoBench/hard_train.json
```

四个桶，判据是 `p = 累计答对数 / 累计 rollout 数`：

| 标签 | 判据 | GRPO 行为 | 含义 |
|---|---|---|---|
| 太易·零梯度 | `p = 1` | `std=0` → 跳过 | 全对，训了等于没训 |
| 偏易 | `0.5 ≤ p < 1` | 有梯度 | 会做但不稳 |
| 偏难 | `0 < p < 0.5` | 有梯度 | **最有价值** |
| 太难·零梯度 | `p = 0` | `std=0` → 跳过 | 一条都没采到对的，RL 够不着 |

分界线不是难易而是**有没有梯度**：只有 `0 < p < 1` 组内 reward 才有方差，两头都被
`compute_loss` 直接跳过。0.5 那条线只是方便看，没有机制上的意义。

⚠ **`--max_epoch 1.0`**：`p` 是跨 epoch 混算的，而策略一直在变。模型真学到东西的话，
后面 epoch 正确率天然更高，混算会把题目**显得比实际简单**。做筛选只用第一遍的数据。
代价是每题只有 8 条 rollout，`p` 只能取 9 个离散值。

⚠ **`太难·零梯度` 不能直接当难题用。** 脚本会额外算"是不是稳定错成同一个选项"：是 →
模型有确定的错误信念、或该题 GT 可疑，RL 靠采样纠正不了（8 条里一条对的都没有，
advantage 恒为 0），得走 SFT 或人工抽查标注。所以 `--emit_hard` 默认剔除这一桶，
要留得显式加 `--keep_allwrong`。

⚠ **别直接拿难题子集替换全量训练集。** 把简单题全删了，模型可能在那些题上退化。
稳妥做法是难题**过采样**：难题复制 2~3 份再和原数据拼起来，让有梯度的组占比上去，
同时留着简单题做锚。

---

## 附一：已知限制

调查过程中发现的、与"能否跑出效果"相关的设计限制，供调参时参考：

- ~~**UrbanVideo 上不了多卡**~~ —— **已于 2026-08-17 解决**，现在 8 卡可跑（跑满 1 epoch 峰值 35.7 GB）。
  这条原来的记录（"单卡 28.3 GB 可跑，8 卡稳定 OOM"，以及一串"已试过且无效"的办法）
  **几乎每一句都是错的**，误导了后续排查好几个小时。完整的错因分析、当时是怎么误判的、
  以及最终解法见[附三](#附三显存排查的四个坑2026-08-17)。
- **UrbanVideo 的显存余量只有约 3.8 GB**（AirCop 是 18 GB）。它靠四项改动叠加才跑通，
  其中 `max_points: 16384` 是有代价的：本数据集 p50=26.2k 点，一半以上样本被下采样，
  输入保真度低于 SFT 时。要拿回精度得上 DeepSpeed ZeRO-2 腾出静态基线（未做）。
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


### 第三轮：UrbanVideo 显存问题（2026-08-17）

| 文件 | 改动 |
|---|---|
| `grpo_trainer.py` | **`compute_loss` 里把 ref 前向移到 policy 带梯度前向之前**（标记 `(4a)`）。原顺序下 ref 的瞬时激活叠在 policy 那张还没 backward 的反向图上面，两个峰值重叠。ref 是冻结的、`ref_logp` 不参与求导，所以调换顺序**数学完全等价**，纯赚显存 |
| `grpo_trainer.py` | **新增 `_wrap_model` 覆盖，打开 DDP 的 `gradient_as_bucket_view`**。DDP 默认同时持有「梯度本体」和「allreduce 通信桶」两份完整拷贝（1.83B 参数 bf16 各 3.7 GB）。HF 的 `TrainingArguments` 没暴露这个开关，只能在 `_wrap_model()` 之后补设——写在 `__init__` 里会**静默失效**（那时 `ddp_handler` 还不存在，而且随后会被 `_wrap_model` 整个覆盖赋值） |
| `grpo_trainer.py` | **`grpo/mem_gb` 改用 `max_memory_allocated()` + 每步 `reset_peak_memory_stats()`**，并新增 `grpo/mem_now_gb`（常驻）。原来只报 `memory_allocated()`（步末常驻），测不到步内瞬时峰值——这是导致本轮多次误判的直接原因 |
| `config_urbanvideo.yaml` | `gradient_checkpointing: false→true`、`max_points: 65536→16384`、`logp_batch_size: 4→1`、新增 `optim: adamw_bnb_8bit`；`save_steps: 200→25`；header 全部重写 |
| `config_urbanvideo.yaml` | **`save_steps: 200→25`**：8 卡下 UrbanVideo 只有 128 步/epoch，200 步意味着一个中间 ckpt 都存不下 |

### 第四轮：留出集 / 难度打标 / 评测重采样 bug（2026-08-17）

| 文件 | 改动 |
|---|---|
| `split_holdout.py` | **新增**：按点云分组 + 按来源分层切留出集，断言两边点云交集为 0。seed 42 确定性，随时可重现 |
| `config_urbanvideo_holdout.yaml` | **新增**：`_train.json` + `lr 1e-6→5e-6` + `epochs 1→5`（510 步）+ `save_steps 125 / limit 4`。lr 和 epoch 的依据见 §十 首轮实测：上一轮 `‖ΔW‖/‖W‖` 只有 1e-5，权重根本没动 |
| `config_urbanvideo_eval_test.yaml` | **新增**：只给 `eval_mcq.py` 用，`train_json` 指向 `_test.json`（字段名沿用是为了复用 `FloodnetGRPODataset`，不代表它是训练数据） |
| `spatiallm_grpo_utils.py` | **修 bug**：`load_point_cloud_tensor` 加 `sample_seed`。原来 `max_points` 封顶时用全局 torch RNG 抽点，而 `eval_mcq.py` 从不调 `torch.manual_seed`——**base 在卡 0、ckpt 在卡 1-3 并行评测时，同一个文件各自抽到不同的 16384 个点**，两边其实在回答不同的点云，对比里混进了纯重采样噪声。UrbanVideo 实测 62% 的点云会触发封顶。给定 seed 后抽样只由 `(seed, path)` 决定。**必须用 `zlib.crc32` 而不是内置 `hash()`**：后者对 str 按进程随机加盐（`PYTHONHASHSEED`），跨进程对不上，改了等于没改 |
| `eval_mcq.py` | `run_one` 接 `sample_seed` 并传给 `get_pcd`，调用处传 `args.seed`；`--seed` 的 help 补上"同时决定抽哪些点"。⚠ seed 一改绝对数字就和旧结果不可比，base 必须和 ckpt 一起重跑 |
| `dataset.py` / `grpo_trainer.py` | 样本返回值加 `idx`，collate 透传（`b.get("idx", -1)` 兼容旧格式） |
| `grpo_trainer.py` | 新增 `difficulty_log`，见 §十一。记录点写在 `std<=1e-6` 的 `continue` **之前** |
| `analyze_difficulty.py` | **新增**：聚合难度标记 → 四桶分布 + 系统性错误检测 + `--emit_hard` 导出 |
| `train_grpo.py` | 透传 `difficulty_log` |

⚠ 训练侧仍走 `sample_seed=None`（全局 RNG）：多个 epoch 里同一场景换着子集看，算一点弱增强。
只有评测必须确定性。要让训练也确定性，`grpo_trainer.py:187` 加个参数即可。

---

## 附三：显存排查的四个坑（2026-08-17）

> 这一节记的是**排查过程中犯的错**，不是结论。留着是因为这些错误模式会重复出现，
> 而且其中三个都是"看起来已经验证过了、实际上验证方式是错的"。

### 症状

UrbanVideo 8 卡 OOM。当时 README 和 config 里记着一串"已试过且无效"的办法
（降 `max_points`、降 `G`、8bit 优化器、梯度检查点、`expandable_segments`），
并给出结论：**只能单卡跑**。

### 坑 1：用常驻显存去测瞬时开销 —— 四条"无效"结论里有三条因此站不住

`grpo/mem_gb` 当时用的是 `torch.cuda.memory_allocated()`，在**步末**采样，测的是
「当前常驻」。而 OOM 是由**步内瞬时峰值**决定的，点云编码器和 policy 反向图的开销
恰恰全是瞬时的——常驻值根本反映不出来。

后果是：改了 `max_points` 之后去看 `mem_gb`，两组数字几乎逐位相同，于是判定"无效"。
把指标换成 `max_memory_allocated()` + 每步 `reset_peak_memory_stats()` 之后重测，
同一批数据、同一个 seed、第 9 步：

```
max_points=65536: 12.5 21.3 27.1 24.7 24.1 25.2 33.7 26.6 ✗OOM
max_points=16384: 11.7 20.2 25.0 24.6 23.3 24.6 29.4 25.8 35.6 23.4 ... ✓跑满 15 步
```

**差别是决定性的。** 一个"已验证无效"的手段，其实是救命的那一项。

> **教训**：显存指标必须报**峰值**。只报常驻，等于给自己一条永远平坦的曲线，
> 而真正会杀死训练的那个尖峰完全看不见。

### 坑 2：在崩溃点之前的资源上做优化 —— "8bit 优化器无效"

`adamw_bnb_8bit` 理论上能把 AdamW 的 fp32 m/v 从 14.6 GB 压到 3.7 GB，
但实测"完全没用，峰值纹丝不动"。

真实原因：当时 8 卡是在 **step 1 的生成阶段**就爆的，`mem_gb` 一条日志都没打出来。
而**优化器状态是第一次 `optimizer.step()` 才惰性分配的**——崩溃发生时它压根还不在显存里，
砍一个还不存在的东西当然没有任何效果。

等梯度检查点把崩溃点推后到第 4 步之后，8bit 立刻就省出了约 2 GB。

> **教训**：OOM 时先看**栈**，确认爆在哪个阶段，再决定砍什么。
> 「这个东西很大」不等于「爆的时候它在场」。

### 坑 3：靠显存账推理，而不是看 traceback

我先后给出过两个听起来都很合理的错误归因：

1. "是优化器状态 + DDP 梯度桶把 40 GB 顶穿了" —— 账算得很整齐（单卡 28.3 / 4 卡 36.0，
   差 7.7 GB 正好等于梯度本体 + 通信桶两份 bf16 拷贝），但方向是错的。
2. "是 UrbanVideo 点云文件大 30 倍，Sonata 激活撑爆了" —— 也错。

真去看 traceback，位置是固定的：

```
compute_loss → _logprobs_chunked → _seq_logprobs
```

即 **policy 的带梯度 log-prob 前向**，不是 generate，不是点云塔。
看到栈之后五分钟就定位对了，而在此之前推理了几个小时。

> **教训**：traceback 是免费的、确定性的证据。推理是有成本的、容易自洽的猜测。
> 先看栈。

### 坑 4：短测试给出假的"能跑"结论

"UrbanVideo 单卡可跑，峰值 28.3 GB" 这个结论是只跑了三五步得出的。
实际跑到 10 步：

```
10.8 / 14.5 / 35.5 / 18.1 / 33.8 / 21.7 / ✗OOM(第 7 步)
```

差一点就照着这个结论去挂一个 1020 步的通宵训练，几十步内就会挂掉。

显存会随数据波动，而波动的来源是 **GRPO 的零方差跳过**：一组 G 条 rollout
**全对或全错**都会 `std=0` 被 `continue`，那一步不建反向图，显存就低；有对有错才吃梯度。
所以前几步很可能恰好都是被跳过的轻量步。

> **教训**：显存类结论至少跑 15 步。并且留余量要按
> **"所有 group 都产生梯度"的最坏情况**算，不能指望跳过帮你省显存。
>
> 后续补充：15 步也还是不够。跑满 128 步后真峰值是 **35.7 GB**，比 15 步见到的
> 34.4 GB 又高了 1.3 GB，余量从 5 GB 缩到 3.8 GB。这条坑的正确说法是
> **"只有跑满一个 epoch 的峰值才算数"**。

### 一个次要发现：HF 的 `ddp_handler` 会被覆盖

想给 DDP 打开 `gradient_as_bucket_view`（`TrainingArguments` 没暴露），
自然想法是在 `Trainer.__init__` 之后设 `self.accelerator.ddp_handler.gradient_as_bucket_view = True`。
**这会静默失效**：`ddp_handler` 是在 `Trainer._wrap_model()` 里现场 `new` 出来并
**整体覆盖赋值**的（`transformers/trainer.py` 约 2097 行），`__init__` 时它还不存在。
正确做法是覆盖 `_wrap_model`，在 `super()._wrap_model()` 之后再设。

因为这个，我有一整轮测试是白做的——代码看着改了，实际一行没生效。

### 最终解法

四项叠加，**单独去掉任何一项都会重新 OOM**：

| 改动 | 性质 | 攻击的是什么 |
|---|---|---|
| `gradient_checkpointing: true` | 配置 | 语言塔反向激活（最大的一刀，峰值 35→22） |
| `max_points: 65536 → 16384` | 配置 | 点云塔瞬时峰值（第 9 步的救命项） |
| `logp_batch_size: 4 → 1` | 配置 | 单次带梯度前向的瞬时峰值 |
| ref 前向移到 policy 之前 | 代码 | ref 激活与 policy 反向图的**峰值重叠** |
| `optim: adamw_bnb_8bit` | 配置 | 优化器状态（约 2 GB） |
| `gradient_as_bucket_view` | 代码 | DDP 的一份梯度拷贝 |

结果：8 卡跑满 1 个 epoch（128 步）无 OOM，峰值 **35.7 GB** / 39.5 GB。
（15 步冒烟只见到 34.4 GB —— 这个尖峰跑满才出现，见坑 4。）

### 代价与后续

`max_points: 16384` **会改变训练输入**：本数据集 p50=26.2k 点、p90=59.1k，
一半以上样本被随机下采样，保真度低于 SFT 时。会不会掉点需要实测对比。

想拿回精度的两条路（都未做）：

1. 试 `max_points: 32768`——现在其他几项已经腾出余量，可能塞得下。
2. **DeepSpeed ZeRO-2**：分片梯度和优化器状态，每卡静态基线约省 16 GB。
   注意它**不分片激活**，而本例爆的正是激活，所以它是靠压低基线间接买余量。
   接入前必须先改 `grpo_trainer.py` 里 `base = model.module if hasattr(model, "module") else model`
   这个写法——它绕过了外层包装器，在 ZeRO 下会导致 reduce-scatter 不完整，
   **不报错但训错**，比 OOM 难发现得多。

### 一句话总结

四个坑里有三个是**测量方式错了**，不是判断力不够。
显存问题上，先把「量得准」解决掉（报峰值、看栈、跑够步数），
再谈「改什么」——否则每一次"已验证无效"都在给后面的人埋雷。
