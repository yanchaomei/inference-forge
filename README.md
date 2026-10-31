# InferenceForge

**为具体模型、硬件和场景，生成可独立运行的专用推理引擎。**

InferenceForge builds specialized inference engines for a specific model, hardware and workload. Coding agents construct and optimize the runtime with feedback from real execution; the resulting engine runs without an agent in its inference path.

当前版本包含 Qwen2 自动构造示例，以及 Qwen3-8B 主机状态恢复和多会话实验。Coding Agent 生成与修改代码，远程 A40 执行固定验收；适合研究和 AI 基础设施团队探索模型、设备、负载专用化。

我们的交付单位是一套可部署的引擎：源码、启动方式、支持范围与同条件实测结果。构造环境可以复用；生成结果可以采用不同的算子、KV 布局、批处理和内存策略。设计取舍见 [特化引擎路线](docs/SPECIALIZATION.md)。

## 引擎与目标

| 模型 / 硬件 / 场景 | 当前状态 | 入口 |
| --- | --- | --- |
| Qwen2.5-0.5B / 单 A40 / 短请求并发 | 已生成、验收及测量的开发示例 | [运行生成引擎](engines/README.md) |
| Qwen3-8B / 单 A40 / BF16 状态恢复与多会话 | 实验性运行时、实际生成组件及逐输出状态检查；尚无成熟框架优势 | [8B实验入口](experiments/qwen3_8b/README.md) |

先做完整的 8B 引擎及强基线对照，再扩展长上下文、MoE 和其他硬件。构造方法对比 VibeServe、MetaInfer；生成产物的推理性能对比 vLLM、SGLang。这是两个不同问题，分别报告。

```text
任务配置 → Coding Agent → 候选引擎 → 独立 GPU 评测
              ↑                        │
              └──── 结果与失败反馈 ────┘
```

## 已实现

- 真实 Qwen2 数学、权重装载、每请求 KV、prefill/decode、HTTP/SSE 服务。
- 实际外部 Coding Agent 调用、候选文件隔离、JSONL 记录和 token usage。
- 统一远程参考、数值与缓存比较、流式文字及 token 检查、真实请求交错检查。
- 预热和重复测量、错误留档、受限外部请求轨迹缩小。
- 自动生成的批量解码示例，见 `examples/generated/batched/`。
- 8B 主机 KV 保存/恢复，以及实际生成的配套存储与会话消费者；含四格交叉消融、显存预算与完整 logits 检查。

## 已测结果

2026-10-30，Qwen2.5-0.5B-Instruct / 单 A40 / float32 / 4 并发请求、每请求16 token。一次真实 Agent 调用生成批量解码实现，耗时约77.7秒，固定验收通过。按 A-B-B-A 顺序执行，每次含预热和3次测量：

| 实现 | 第一次中位吞吐 | 第二次中位吞吐 |
| --- | ---: | ---: |
| 本项目初版逐请求运行时 | 63.92 token/s | 64.16 token/s |
| Agent 生成的批量解码候选 | 150.82 token/s | 148.80 token/s |

本次小负载配对比值几何平均约 **2.34×**。它是相对自身初版的开发对照；尚未证明优于 vLLM、SGLang、VibeServe，也不是生产 SLO 或大模型结果。详见 [实测数据](results/campaign-20261030.json) 与 [实验说明](docs/EXPERIMENTS.md)。

2026-10-31 的8B实验中，锁页主机恢复组件相对自身初版快2.78–3.87倍，但恢复后16token总时间仅降低0.4%–4.6%。多会话联合版本的开发中位提升仅约0.815%，尚不足以证明稳健优势。完整[原始逐请求记录](results/qwen3-8b-sessions-20261031.json)与[复现范围](experiments/qwen3_8b/README.md)同时公开。

## 快速开始

需要本地 Python、已安装并登录的 Codex CLI，以及可 SSH 访问、安装 tmux 的 Linux GPU 服务器。代码编辑与生成在本地；模型执行在服务器。凭据不上传到 GPU 服务器。

```bash
git clone https://github.com/yanchaomei/inference-forge.git
cd inference-forge
python -m pip install -e .
```

在远程的独立环境安装推理依赖：

```bash
python -m pip install -e '.[inference,test]'
```

从 [Qwen 官方模型页](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)准备模型。仓库不分发权重。任务记录了本次实测权重 SHA256；若使用不同权重版本，请先固定并验证该版本。

在仓库根目录配置本地控制器使用的远程路径和设备：

```bash
export FORGE_MODEL_PATH=/path/on/gpu-server/Qwen2.5-0.5B-Instruct
export FORGE_GPU_UUID=GPU-your-device-uuid

inference-forge --task tasks/qwen25_05b/task.json \
  --output runs/example-campaign --host user@gpu-server --port 22 \
  --key ~/.ssh/id_ed25519 --remote-root /path/on/gpu-server/inference-forge \
  --remote-python /path/to/venv/bin/python --rounds 1 \
  --generator-model YOUR_AVAILABLE_CODEX_MODEL --clean-generator-config
```

运行器先测种子引擎，再运行生成器与固定验收。请指定账户实际可用的模型。`runs/` 保存提示词、原始生成事件、usage、候选快照、远程任务身份和结果。中断时先查看已记录的远程会话，避免重复启动。

只验收现有引擎，可在 GPU 服务器的仓库根目录运行：

```bash
forge-pilot --task tasks/qwen25_05b/task.json --output results/raw/local-check
```

## 验证与范围

在 GPU 环境设好模型路径后运行：

```bash
export CUDA_VISIBLE_DEVICES=$FORGE_GPU_UUID
python -m pytest tests -q
```

首页快速开始仍对应短上下文 Qwen2.5-0.5B，服务上限512 token、固定长度greedy输出。8B实验采用独立依赖环境和入口，见[实验说明](experiments/qwen3_8b/README.md)。完整8B构造方法对照、多硬件、独立留出与生产服务性能仍在推进。

8B 准备工具在 [scripts/prepare_model.py](scripts/prepare_model.py)：消费目标配置，在 GPU 服务器下载指定 revision，并用官方记录校验全部分片、配置和 tokenizer。它只完成模型准备，不代表构造或推理验收成功。

## 参与与来源

代码采用 [MIT](LICENSE)。见 [贡献指南](CONTRIBUTING.md)、[依赖与研究来源](NOTICE.md)、[系统说明](docs/ARCHITECTURE.md)。欢迎提交可复现失败和同条件性能改进，不需要先提交复杂架构。
