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

实测范围与原始数值见 [运行记录](../results/campaign-20261030.json)。Qwen3-8B 仍在准备，不提供占位引擎或虚构速度。
