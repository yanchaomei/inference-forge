# 生成引擎

这里是可运行产物的入口。它们在推理时不调用构造 Agent。

## Qwen2.5-0.5B / A40 / 短请求批量解码

源码在 [examples/generated/batched](../examples/generated/batched/)，源于真实 C2 构造运行。它只支持 Qwen2、float32、最多 512 token 上下文和 64 token 输出；不是 8B 引擎。

在 GPU 主机准备模型及仓库的 inference 依赖后，在仓库根目录启动：

```bash
CUDA_VISIBLE_DEVICES=GPU-YOUR-UUID \
python -m examples.generated.batched.server \
  --model /path/to/Qwen2.5-0.5B-Instruct --dtype float32 \
  --port 8000 --event-log runs/engine-events.jsonl
```

该模块使用标准 Python namespace package 从仓库源码启动。HTTP 入口为 `127.0.0.1:8000/v1/completions`，支持本项目固定长度 greedy 请求和 SSE，不宣称完整 OpenAI 协议兼容。

实测范围与原始数值见 [运行记录](../results/campaign-20261030.json)。

## Qwen3-8B / A40 / 状态恢复与多会话实验

新增 [主机KV组件](qwen3_8b_kv/README.md)与[配套会话运行时](qwen3_8b_sessions/README.md)。它们在真实8B权重上验证过状态保持与显存预算，但仍为实验性API，没有完整HTTP服务或成熟框架性能优势。依赖、生成和评测方式见[统一实验入口](../experiments/qwen3_8b/README.md)。
