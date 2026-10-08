# OLMo 3 Think 的 SFT 示例

本示例用于在 ms-swift 中微调 `allenai/Olmo-3-7B-Think`，对应 `--model_type olmo3 --template olmo3_thinking`。模型使用 Transformers 原生的 `Olmo3ForCausalLM`；直接读取原模型配置，保留 `layer_types` 中的滑动窗口/全局注意力、`sliding_window` 和 YaRN RoPE 参数，无需复制模型实现或修改 `config.json`。

## 安装与模型准备

建议使用 Linux 或 WSL2 的 NVIDIA CUDA 训练环境，先安装与机器 CUDA 环境匹配的 PyTorch，再在**包含本适配修改的 ms-swift 仓库根目录**执行：

```bash
python -m pip install -e .
python -m pip install 'transformers>=4.57.0,<5.17.0'
```

OLMo 3 需要 `transformers>=4.57.0`；上面的上限遵循当前仓库依赖约束。全参数多卡示例还需要与 PyTorch/CUDA 匹配的 DeepSpeed，可按当前仓库约束安装：

```bash
python -m pip install 'deepspeed<0.19'
```

`MODEL` 可以是 Hugging Face 模型 ID，也可以是完整模型的本地目录。脚本默认 `--use_hf true`，使用模型 ID 时需要能够访问 Hugging Face。离线训练应提前准备完整模型目录，包括模型权重及分片索引（如有）、`config.json`、tokenizer 文件和 tokenizer 配置等。**只有 `D:\code\olmo\config.json` 不能加载模型进行训练。** 若 `D:\code\olmo` 已包含完整模型，可以直接将该目录作为 `--model`。

## 数据格式

`train.jsonl` 提供 4 条教学样本，只用于检查数据流程，应替换为实际业务数据，并单独准备验证集。每行是一个 JSON 对象，使用 ms-swift 的 `messages` 格式，支持单轮、多轮以及可选的 `system` 消息：

```json
{"messages":[{"role":"user","content":"计算 18 × 7。"},{"role":"assistant","content":"<think>10 × 7 = 70，8 × 7 = 56，相加得到 126。</think>\n18 × 7 = 126。"}]}
```

训练 assistant 的 `content` 应包含完整的 `<think>推理内容</think>最终回答`。使用 `--preserve_thinking true`，保留多轮历史 assistant 中已有的推理内容。若原始数据只有最终回答，按原样提供；模板不会自动伪造推理，持续使用大量此类数据可能改变模型的推理输出习惯。不要在数据中手动拼接 `<|im_start|>` 等角色边界，也不要新增 special tokens 或扩充词表。

本适配针对普通 `system/user/assistant` 对话 SFT。格式以模型仓库的实际 [chat_template.jinja](https://huggingface.co/allenai/Olmo-3-7B-Think/blob/main/chat_template.jinja) 为准：system 会追加无可用函数的说明，历史 assistant 以 `<|im_end|>\n` 分隔，最后一轮以 `<|endoftext|>` 结束；推理提示以 `<think>` 结尾，不额外加换行。模型卡中的简化 system 示例与该文件不同。

默认只计算 assistant 部分的训练损失，推理内容和最终回答都参与 SFT。示例使用 `max_length=2048`；实际数据应根据完整对话长度调整，避免截断推理与最终回答。模型原始上下文配置不代表当前硬件能以该长度训练。

## Linux / WSL2 训练

在仓库根目录运行 LoRA 示例：

```bash
bash examples/train/think_model/olmo3_thinking/sft_lora.sh
```

使用本地模型及自己的数据：

```bash
MODEL=/models/Olmo-3-7B-Think \
DATASET=/data/train.jsonl \
OUTPUT_DIR=output/olmo3-custom-lora \
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
bash examples/train/think_model/olmo3_thinking/sft_lora.sh
```

LoRA 显式适配 `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`。脚本默认 bf16、SDPA、梯度检查点，关闭 packing、padding-free 与 Liger。bf16 需要硬件支持；脚本没有启用量化或 Unsloth。

全参数训练默认使用 4 张 GPU 和 DeepSpeed ZeRO-3：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash examples/train/think_model/olmo3_thinking/sft_full.sh
```

全参数训练还需要保存梯度、优化器状态和激活值，对显存、主存及磁盘空间要求明显更高。4 卡是启动配置示例，不能据此保证显存足够；根据实际 GPU、序列长度和批大小评估资源。`NPROC_PER_NODE` 应等于当前节点使用的 GPU 数。显存充足时可按需使用 `--deepspeed zero2`。两个脚本都支持追加 CLI 参数，例如：

```bash
bash examples/train/think_model/olmo3_thinking/sft_lora.sh \
  --max_length 4096 --val_dataset /data/val.jsonl --eval_strategy steps --eval_steps 100
```

默认不从这 4 条演示数据划分验证集。实际训练请增加高质量验证集，并根据效果调整学习率、训练轮数与有效批大小：`每卡批大小 × GPU 数 × gradient_accumulation_steps`。

## Windows PowerShell 单卡命令

本机可直接使用 ms-swift CLI，避免执行 Bash 脚本。先在本适配仓库根目录完成安装，并确认本机 PyTorch CUDA 可用及 bf16 硬件支持。多卡与 DeepSpeed 训练建议使用 Linux / WSL2。

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
Remove-Item Env:NPROC_PER_NODE -ErrorAction SilentlyContinue
swift sft `
  --model 'D:\code\olmo' `
  --model_type olmo3 --template olmo3_thinking --use_hf true `
  --tuner_type lora `
  --dataset 'examples/train/think_model/olmo3_thinking/train.jsonl' `
  --split_dataset_ratio 0 --eval_strategy no `
  --torch_dtype bfloat16 --attn_impl sdpa `
  --gradient_checkpointing true --packing false --padding_free false `
  --preserve_thinking true --use_liger_kernel false `
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj `
  --lora_rank 8 --lora_alpha 32 --learning_rate 1e-4 `
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 `
  --num_train_epochs 1 --max_length 2048 --warmup_ratio 0.03 `
  --logging_steps 1 --save_steps 100 --save_total_limit 2 `
  --dataloader_num_workers 0 --load_from_cache_file false --report_to none `
  --output_dir 'output/olmo3-7b-think-lora'
```

这里的 `D:\code\olmo` 必须是完整模型目录。也可替换为 `'allenai/Olmo-3-7B-Think'` 在线加载。

## 推理与验证

LoRA 训练后将以下 adapter 路径替换为训练日志中实际生成的 checkpoint。ms-swift 默认会在 `output_dir` 下添加版本及时间子目录。

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
  --model allenai/Olmo-3-7B-Think \
  --model_type olmo3 --template olmo3_thinking --use_hf true \
  --adapters output/olmo3-7b-think-lora/v0-YYYYMMDD-HHMMSS/checkpoint-N \
  --infer_backend transformers --torch_dtype bfloat16 --attn_impl sdpa \
  --preserve_thinking true --max_new_tokens 2048 --stream true
```

全参数训练后将 `--model` 改为实际的完整模型 checkpoint，并移除 `--adapters`。若在当前 shell 中运行过多卡训练，推理前清除 `NPROC_PER_NODE`（Bash 用 `unset NPROC_PER_NODE`，PowerShell 用 `Remove-Item Env:NPROC_PER_NODE`）。推理生成长度需要为推理内容和最终回答都留出空间。

适配回归测试在仓库根目录执行：

```bash
python -m pytest tests/general/test_olmo3.py -q
```

这些测试用于检查模型注册、聊天模板、思考内容保留与小规模模型训练兼容性。适配已在 Python 3.12、Transformers 4.57.6、PEFT 0.17.1 的 CPU 环境通过 11 项离线回归测试，并用本地随机初始化的两层 OLMo 3 完成 `swift sft` 单步 LoRA 训练（SDPA、梯度检查点）。尚未执行真实 7B 模型的权重加载、GPU SFT、收敛或显存验证；小模型测试不能替代目标环境中的真实模型验证。

参考：[模型与模型文件](https://huggingface.co/allenai/Olmo-3-7B-Think)、[Transformers OLMo 3 文档](https://huggingface.co/docs/transformers/model_doc/olmo3)。
