# SOD: Step-wise On-policy Distillation for Small Language Model Agents

## 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型下载](#模型下载)
- [Sandbox 配置](#sandbox-配置)
- [训练流程](#训练流程)
  - [Step 1: Cold-Start SFT](#step-1-cold-start-sft)
  - [Step 2: SOD 训练](#step-2-sod-训练)
- [评估](#评估)
- [常见问题](#常见问题)

---

## 项目简介

SOD（Step-wise On-policy Distillation）是一种用于小模型智能体的蒸馏方法。通过自适应的步级加权机制，将大模型（Teacher, 4B）的能力蒸馏到小模型（Student, 0.6B/1.7B）中，在 AIME、GPQA、LiveCodeBench 等数学/代码/科学推理任务上取得显著提升。

**关键特点**：
- 解决工具调用蒸馏中的级联错误传播问题
- 在计算开销几乎不增加的情况下提升蒸馏效果
- 0.6B 模型在 AIME 2025 上达到 26.13% average@32

---

## 环境配置

### 硬件要求

| 阶段 | GPU 需求 |
|------|----------|
| SFT 训练 | 8× GPU（推荐 H20 96GB） |
| SOD 训练 | 8× GPU（推荐 H20 96GB） |
| 评估 | 1~8× GPU |

### 创建 Conda 环境

> ⚠️ **必须使用 Python 3.11**，脚本中的 flash-attention 和 flashinfer 的预编译 wheel 是专为 `cp311` 构建的。

```bash
conda create -n SOD python=3.11 -y
conda activate SOD
```

### 安装依赖

```bash
cd /cfs_cloud_code/jackxlyan/SOD

# 运行官方安装脚本（包含 vllm、torch、flash-attention、ray 等所有依赖）
source scripts/install_vllm_sglang_mcore.sh

# 以可编辑模式安装项目本体
pip install -e .[vllm]
```

安装脚本会自动安装以下核心组件：
1. **PyTorch 2.6.0** + CUDA 12.4
2. **vLLM 0.8.5.post1** — 推理加速
3. **FlashAttention 2.7.4** — 高效注意力
4. **FlashInfer 0.2.2** — KV-cache 优化
5. **Ray** — 分布式训练调度
6. **transformers, accelerate, datasets** — HuggingFace 生态
7. **wandb** — 实验追踪
8. **hydra-core** — 配置管理

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "import verl; print('verl OK')"
```

---

## 数据准备

### 下载数据集

数据集托管在 HuggingFace 上，共 3 个：

| 数据集 | 用途 | 大小约 |
|--------|------|--------|
| [Open-AgentRL-SFT-3K](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K) | Step 1: Cold-start SFT | ~8 MB |
| [Open-AgentRL-30K](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K) | Step 2: SOD 训练 | ~175 MB |
| [Open-AgentRL-Eval](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-Eval) | 评估（AIME/GPQA/LCB） | ~250 KB |

### 下载命令

```bash
# 创建数据目录
mkdir -p /jizhicfs/jackxlyan/dataset/sod_data
cd /jizhicfs/jackxlyan/dataset/sod_data

# 如果 HuggingFace 直连不通，先设镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载 SFT 3K 数据
hf download Gen-Verse/Open-AgentRL-SFT-3K --repo-type dataset --local-dir ./sft_3k

# 下载 RL 30K 数据
hf download Gen-Verse/Open-AgentRL-30K --repo-type dataset --local-dir ./rl_30k

# 下载评估数据
hf download Gen-Verse/Open-AgentRL-Eval --repo-type dataset --local-dir ./eval
```

> **注意**：如果 `hf` 命令不存在，请安装：`pip install huggingface_hub[cli]`
>
> 如果你的 `huggingface-cli` 版本较老，请用 `huggingface-cli download` 替代 `hf download`。

### 验证数据

下载完成后，目录结构应如下：

```
/jizhicfs/jackxlyan/dataset/sod_data/
├── sft_3k/
│   └── full_sft_3k_shuffled_v4.parquet      # SFT 训练数据
├── rl_30k/
│   └── Open-AgentRL-30K.parquet             # SOD/RL 训练数据
└── eval/
    ├── aime2024/
    │   └── aime_2024_problems.parquet       # AIME 2024 评估
    ├── aime2025/
    │   └── aime_2025_problems.parquet       # AIME 2025 评估
    ├── gpqa-diamond/
    │   └── gpqa_diamond.parquet             # GPQA-Diamond 评估
    └── livecodebench-v6/
        └── lcb_v6_2502_2505.parquet         # LiveCodeBench v6 评估
```

用 Python 快速验证：

```bash
python -c "
import pandas as pd
paths = {
    'SFT 3K': '/jizhicfs/jackxlyan/dataset/sod_data/sft_3k/full_sft_3k_shuffled_v4.parquet',
    'RL 30K': '/jizhicfs/jackxlyan/dataset/sod_data/rl_30k/Open-AgentRL-30K.parquet',
    'AIME 2024': '/jizhicfs/jackxlyan/dataset/sod_data/eval/aime2024/aime_2024_problems.parquet',
    'AIME 2025': '/jizhicfs/jackxlyan/dataset/sod_data/eval/aime2025/aime_2025_problems.parquet',
}
for name, path in paths.items():
    try:
        df = pd.read_parquet(path)
        print(f'✅ {name}: {len(df)} rows')
    except Exception as e:
        print(f'❌ {name}: {e}')
"
```

---

## 模型下载

### 需要的模型

| 模型 | 用途 | 链接 |
|------|------|------|
| Qwen3-1.7B（或 Qwen3-0.6B） | Student 基座模型（SFT 起点） | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| SOD-GRPO_teacher-4B | Teacher 模型（SOD 蒸馏用） | [youngzhong/SOD-GRPO_teacher-4B](https://huggingface.co/youngzhong/SOD-GRPO_teacher-4B) |

### 下载命令

```bash
mkdir -p /jizhicfs/jackxlyan/pretrained/qwen3
cd /jizhicfs/jackxlyan/pretrained/qwen3

# 如果需要镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Student 基座模型
hf download Qwen/Qwen3-1.7B --local-dir ./Qwen3-1.7B

# 下载 Teacher 模型
hf download youngzhong/SOD-GRPO_teacher-4B --local-dir ./SOD-GRPO_teacher-4B
```

### 验证模型

```bash
ls /jizhicfs/jackxlyan/pretrained/qwen3/Qwen3-1.7B/
# 应包含: config.json, model-00001-of-00002.safetensors, model-00002-of-00002.safetensors, tokenizer.json 等

ls /jizhicfs/jackxlyan/pretrained/qwen3/SOD-GRPO_teacher-4B/
# 应包含: config.json, model*.safetensors, tokenizer.json 等
```

---

## Sandbox 配置

SOD 训练和评估中涉及代码执行（LiveCodeBench 等），需要一个代码沙箱服务。

### 方案 A：使用本地 Sandbox（推荐，无需 Docker）

项目已自带一个轻量级本地 Sandbox 服务 `local_sandbox.py`：

#### 启动服务

```bash
cd /cfs_cloud_code/jackxlyan/SOD

# 前台运行（调试用）
python local_sandbox.py

# 或后台运行（训练时推荐）
nohup python local_sandbox.py > sandbox.log 2>&1 &
```

服务默认在 `http://localhost:8080/run_code` 上监听。

#### 验证服务

```bash
curl -X POST http://localhost:8080/run_code \
  -H "Content-Type: application/json" \
  -d '{"code": "print(1+1)", "language": "python"}'
```

应返回：`{"status": "Success", "run_result": {"stdout": "2\n", ...}}`

#### 关闭服务

```bash
# 如果前台运行：Ctrl+C
# 如果后台运行：
ps aux | grep local_sandbox
kill <PID>
```

### 方案 B：使用 SandboxFusion（官方推荐）

参考 [SandboxFusion 部署文档](https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment) 或使用 [火山引擎代码沙箱](https://www.volcengine.com/docs/6662/1539235)。

### 配置 Sandbox URL

确保以下两处配置指向你的 Sandbox 地址（本地默认是 `http://localhost:8080/run_code`）：

1. **`recipe/demystify/sandbox_fusion_tool_config.yaml`**：

```yaml
sandbox_fusion_url: "http://localhost:8080/run_code"
```

2. **`verl/utils/reward_score/livecodebench/code_math.py`** 中的 `sandbox_fusion_url` 参数。

---

## 训练流程

### Step 1: Cold-Start SFT

用 3K 多轮智能体轨迹数据对基座模型进行监督微调，给 Student 一个好的初始化。

#### 配置并运行

```bash
cd /cfs_cloud_code/jackxlyan/SOD
conda activate SOD

# 设置环境变量
export MODEL_PATH=/jizhicfs/jackxlyan/pretrained/qwen3/Qwen3-1.7B
export TRAIN_DATA=/jizhicfs/jackxlyan/dataset/sod_data/sft_3k/full_sft_3k_shuffled_v4.parquet
export SAVE_PATH=./checkpoint/qwen3_sft
export NPROC_PER_NODE=8  # GPU 数量，按实际调整

# 启动 SFT 训练
bash examples/SOD/run_sft.sh
```

#### SFT 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.max_length` | 32768 | 最大序列长度 |
| `data.train_batch_size` | 128 | 全局 batch size |
| `data.micro_batch_size_per_gpu` | 16 | 每卡 micro batch |
| `trainer.total_epochs` | 5 | 训练轮数 |
| `trainer.save_freq` | 50 | 每 50 步保存 checkpoint |
| `ulysses_sequence_parallel_size` | 4 | 序列并行度 |

#### 合并 Checkpoint

SFT 训完后，FSDP checkpoint 需要合并为 HuggingFace 格式：

```bash
# 查看 checkpoint 步数
ls ./checkpoint/qwen3_sft/

# 合并（替换 global_step_xxx 为实际步数）
python3 -m verl.model_merger merge --backend fsdp \
    --local_dir ./checkpoint/qwen3_sft/global_step_xxx \
    --target_dir ./checkpoint/qwen3_sft/global_step_xxx/huggingface
```

合并后的模型在 `./checkpoint/qwen3_sft/global_step_xxx/huggingface/`，将作为 Step 2 的 Student 模型。

---

### Step 2: SOD 训练

Step-wise On-policy Distillation：用 Teacher 模型指导 Student 模型进行蒸馏训练。

#### 前置条件

- ✅ Step 1 SFT 已完成，模型已合并
- ✅ Teacher 模型已下载
- ✅ RL 30K 数据已就绪
- ✅ Sandbox 服务已启动

#### 配置并运行

```bash
cd /cfs_cloud_code/jackxlyan/SOD
conda activate SOD

# 确保 Sandbox 在运行
nohup python local_sandbox.py > sandbox.log 2>&1 &

# 设置环境变量
export STUDENT_MODEL_PATH=./checkpoint/qwen3_sft/global_step_xxx/huggingface
export TEACHER_MODEL_PATH=/jizhicfs/jackxlyan/pretrained/qwen3/SOD-GRPO_teacher-4B
export OPEN_AGENT_RL=/jizhicfs/jackxlyan/dataset/sod_data/rl_30k/Open-AgentRL-30K.parquet
export AIME_2024=/jizhicfs/jackxlyan/dataset/sod_data/eval/aime2024/aime_2024_problems.parquet
export AIME_2025=/jizhicfs/jackxlyan/dataset/sod_data/eval/aime2025/aime_2025_problems.parquet

# 启动 SOD 训练
bash examples/SOD/run_sod.sh
```

#### SOD 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_batch_size` | 64 | 全局 batch size |
| `n_resp_per_prompt` | 16 | 每条 prompt 采样的 response 数 |
| `n_resp_per_prompt_val` | 32 | 评估时每条 prompt 采样数（average@32） |
| `max_turns` | 16 | 最大多轮对话轮数 |
| `max_response_length` | 20480 | 最大 response token 数 |
| `actor_lr` | 1e-6 | Actor 学习率 |
| `stepwise_delta` | 0.2 | 步级权重上界偏移 |
| `stepwise_opd_coef` | 1.0 | OPD 全局系数 |
| `teacher_kl_coef` | 0.002 | Teacher KL 系数 |
| `infer_tp` | 4 | 推理时 Tensor Parallel |
| `train_sp` | 4 | 训练时 Sequence Parallel |

---

## 评估

### AIME 评估

```bash
bash examples/SOD/eval/run_eval_aime.sh
```

### 其他评估

可在 wandb 项目中查看 average@32 / pass@32 / maj@32 指标。

---

## 路径速查表

```bash
# ===== 数据路径 =====
SFT_DATA=/jizhicfs/jackxlyan/dataset/sod_data/sft_3k/full_sft_3k_shuffled_v4.parquet
RL_DATA=/jizhicfs/jackxlyan/dataset/sod_data/rl_30k/Open-AgentRL-30K.parquet
AIME_2024=/jizhicfs/jackxlyan/dataset/sod_data/eval/aime2024/aime_2024_problems.parquet
AIME_2025=/jizhicfs/jackxlyan/dataset/sod_data/eval/aime2025/aime_2025_problems.parquet

# ===== 模型路径 =====
STUDENT_BASE=/jizhicfs/jackxlyan/pretrained/qwen3/Qwen3-1.7B
TEACHER=/jizhicfs/jackxlyan/pretrained/qwen3/SOD-GRPO_teacher-4B

# ===== 项目路径 =====
PROJECT_ROOT=/cfs_cloud_code/jackxlyan/SOD
```

---

## 完整运行流程一键脚本示例

```bash
#!/bin/bash
# === 前置：确保 conda activate SOD ===

cd /cfs_cloud_code/jackxlyan/SOD

# 1. 启动 Sandbox
nohup python local_sandbox.py > sandbox.log 2>&1 &
sleep 2
curl -s http://localhost:8080/ && echo " Sandbox OK"

# 2. Step 1: SFT
export MODEL_PATH=/jizhicfs/jackxlyan/pretrained/qwen3/Qwen3-1.7B
export TRAIN_DATA=/jizhicfs/jackxlyan/dataset/sod_data/sft_3k/full_sft_3k_shuffled_v4.parquet
export SAVE_PATH=./checkpoint/qwen3_sft
export NPROC_PER_NODE=8
bash examples/SOD/run_sft.sh

# 3. 合并 SFT checkpoint（替换 step 数）
python3 -m verl.model_merger merge --backend fsdp \
    --local_dir ./checkpoint/qwen3_sft/global_step_250 \
    --target_dir ./checkpoint/qwen3_sft/global_step_250/huggingface

# 4. Step 2: SOD 训练
export STUDENT_MODEL_PATH=./checkpoint/qwen3_sft/global_step_250/huggingface
export TEACHER_MODEL_PATH=/jizhicfs/jackxlyan/pretrained/qwen3/SOD-GRPO_teacher-4B
export OPEN_AGENT_RL=/jizhicfs/jackxlyan/dataset/sod_data/rl_30k/Open-AgentRL-30K.parquet
export AIME_2024=/jizhicfs/jackxlyan/dataset/sod_data/eval/aime2024/aime_2024_problems.parquet
export AIME_2025=/jizhicfs/jackxlyan/dataset/sod_data/eval/aime2025/aime_2025_problems.parquet
bash examples/SOD/run_sod.sh
```

---

## 引用

```bibtex
@article{zhong2026sod,
  title={SOD: Step-wise On-policy Distillation for Small Language Model Agents},
  author={Zhong, Qiyong and Zheng, Mao and Song, Mingyang and Lin, Xin and Sun, Jie and Jiang, Houcheng and Wang, Xiang and Fang, Junfeng},
  journal={arXiv preprint arXiv:2605.07725},
  year={2026}
}
```
