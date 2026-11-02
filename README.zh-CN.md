# InferenceForge

[![CPU contracts](https://github.com/yanchaomei/inference-forge/actions/workflows/check.yml/badge.svg?branch=main)](https://github.com/yanchaomei/inference-forge/actions/workflows/check.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**为具体模型、GPU 和负载，构造并运行专用推理引擎。**

[English](README.md) · [简体中文](README.zh-CN.md) · [快速开始](docs/QUICKSTART.md) · [API](docs/API.md) · [模型支持](docs/SUPPORTED_MODELS.md) · [参与贡献](CONTRIBUTING.md)

InferenceForge 通过 Coding Agent 和真实执行反馈构造推理引擎。生成的引擎可以独立运行，推理过程不需要 Agent 或模型服务账号。

可以先运行仓库自带的 Qwen2.5-0.5B 引擎，通过 OpenAI Python SDK 接入应用，再使用构造工具开发新候选。项目面向研究和 AI 基础设施开发者，探索模型、硬件和负载专用化。

## 安装并启动

需要 Linux、Python 3.10+ 和 NVIDIA GPU。当前服务目标为 **Qwen2.5-0.5B-Instruct、float32、单 GPU**，历史实测硬件为 A40。输入与输出总长度最多 512 token，每次固定生成 1–64 token，采用 greedy 解码。

目前尚未发布到 PyPI，请从源码安装：

```bash
git clone https://github.com/yanchaomei/inference-forge.git
cd inference-forge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[inference]'
inference-forge doctor --json
```

按 [下载指南](docs/QUICKSTART.md#download-the-model) 准备模型，将本地模型目录传入服务：

```bash
inference-forge serve /path/to/Qwen2.5-0.5B-Instruct \
  --port 8000 --served-model-name qwen2.5-0.5b
```

默认地址为 `http://127.0.0.1:8000`，用 `curl --fail http://127.0.0.1:8000/health` 检查是否就绪。

## 使用 OpenAI SDK

```bash
python -m pip install 'openai>=1.50,<3'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "用一句话解释 KV 缓存。"}],
    max_tokens=32,
    temperature=0,
)
print(response.choices[0].message.content)
```

支持文本补全、纯文本聊天和流式输出。聊天请求使用模型自带模板；参数和流式用法见 [API 文档](docs/API.md)。如需向外提供服务，由已有网关处理鉴权和 TLS。

## 构造新引擎

构造流程从可运行的种子开始，由 Coding Agent 修改候选，再交给独立 GPU 主机执行数值验证与负载测量。结果反馈给下一轮修改。

```text
模型 + 硬件 + 负载 → Coding Agent → 候选引擎 → GPU 评测
                         ↑                       │
                         └────── 执行反馈 ───────┘
```

运行 `inference-forge construct --help` 查看参数，完整环境与命令见 [开发指南](docs/DEVELOPMENT.md#construct-a-candidate)。只有构造新候选时需要配置 Coding Agent；运行已有引擎不需要。

## 模型与实验

| 目标 | 可运行内容 | 入口 |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct / float32 / 单 A40 | 可安装引擎、批量解码、HTTP 与 SSE 服务 | [快速开始](docs/QUICKSTART.md) |
| Qwen3-8B / BF16 / 单 A40 | 独立的 KV 保存恢复与多会话实验 | [8B 实验](experiments/qwen3_8b/README.md) |

Qwen3-8B 尚未接入 `serve`。运行 `inference-forge models --json` 查看服务目标，完整范围见 [支持矩阵](docs/SUPPORTED_MODELS.md)。

在单 A40、float32、4 并发、每请求 16 输出 token 的开发负载中，0.5B 批量解码候选的吞吐约为本项目初版逐请求运行时的 **2.34 倍**。独立 8B 存储与会话实验的三次开发重复中，联合候选相对原组件的总耗时中位数降低 **0.815%**。原始样本与解读见 [0.5B 对照与范围](docs/EXPERIMENTS.md) 和 [8B 探索性结果](experiments/qwen3_8b/README.md#结果与未解决的问题)。

## 文档导航

| 要做什么 | 从这里开始 |
| --- | --- |
| 安装、下载模型、发起请求 | [快速开始](docs/QUICKSTART.md) |
| 接入应用 | [API 文档](docs/API.md) |
| 确认模型、精度与硬件范围 | [模型支持](docs/SUPPORTED_MODELS.md) |
| 修改代码或构造引擎 | [开发指南](docs/DEVELOPMENT.md) |
| 理解设计 | [系统说明](docs/ARCHITECTURE.md) · [特化路线](docs/SPECIALIZATION.md) |
| 了解下一步 | [路线图](docs/ROADMAP.md) |
| 查看发布版本 | [Releases](https://github.com/yanchaomei/inference-forge/releases) · [v0.2.0 说明](docs/RELEASE-v0.2.0.md) |

## 参与贡献

欢迎提交可复现失败、改进示例、数值检查和有实测支持的性能优化。见 [贡献指南](CONTRIBUTING.md)、[Issue 表单](https://github.com/yanchaomei/inference-forge/issues/new/choose) 与 [安全政策](SECURITY.md)。

## 相关材料

- [演示与视频](https://yanchaomei.github.io/inference-forge/)：已录制的 0.5B 构造示例。
- [34 页作品介绍](https://yanchaomei.github.io/inference-forge/project-introduction.pdf) 与 [参赛材料](docs/SUBMISSION.md)。
- [v0.2.0 验证记录](results/release-v0.2.0-validation.json)：该历史版本的 HTTP/SSE、并发与数值检查。

代码采用 [MIT](LICENSE) 许可证。模型权重单独下载并遵循其各自许可。依赖与研究来源见 [NOTICE](NOTICE.md)。
