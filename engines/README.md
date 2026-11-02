# 生成引擎

这里汇总可运行的引擎和实验入口。运行已生成的引擎不需要构造 Agent。

## Qwen2.5-0.5B / A40 / 短请求批量解码

当前服务使用包内的 [Qwen2 数值后端](../forge/backends/qwen2/engine.py) 和 [HTTP/SSE 服务](../forge/serving.py)。数值实现来自真实 C2 构造运行，服务层提供统一 CLI 和 OpenAI SDK 可用的文本接口。支持 Qwen2.5-0.5B-Instruct、float32、输入与输出总长度最多 512 token、固定生成 1–64 token。

在 Linux GPU 主机安装推理依赖并准备模型后启动：

```bash
CUDA_VISIBLE_DEVICES=GPU-YOUR-UUID \
inference-forge serve /path/to/Qwen2.5-0.5B-Instruct \
  --port 8000 --served-model-name qwen2.5-0.5b \
  --event-log runs/engine-events.jsonl
```

服务默认监听 `127.0.0.1:8000`，提供文本补全、纯文本聊天和 SSE。模型下载与客户端示例见 [快速开始](../docs/QUICKSTART.md)，参数范围见 [API 文档](../docs/API.md)。

### 历史源码与当前兼容入口

2026-10-30 实验的两个原始生成文件固定保存在 [v0.2.0 的 examples/generated/batched/](https://github.com/yanchaomei/inference-forge/tree/v0.2.0/examples/generated/batched)。当前同名目录提供兼容导入和模块启动入口，实际执行包内后端与服务，避免维护两份实现。

复现历史测量请使用 v0.2.0 的源码与依赖说明；使用新 CLI 或聊天接口请按当前快速开始安装。性能数据和对照范围见 [实验说明](../docs/EXPERIMENTS.md) 与 [原始运行记录](../results/campaign-20261030.json)。

## Qwen3-8B / A40 / 状态恢复与多会话实验

[主机 KV 组件](qwen3_8b_kv/README.md) 与 [配套会话运行时](qwen3_8b_sessions/README.md) 使用独立的实验 API，在真实 8B 权重上检查状态保持和显存预算。它们尚未接入 `inference-forge serve`。

依赖、生成、评测命令及结果解读见 [8B 实验指南](../experiments/qwen3_8b/README.md)。
