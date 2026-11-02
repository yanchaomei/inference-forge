# 实验说明

本页记录 0.5B 构造与运行实验，历史测量对应冻结的候选源码。当前安装与 API 用法见 [快速开始](QUICKSTART.md)；8B 数据与四格对照见 [独立实验指南](../experiments/qwen3_8b/README.md)。

## Bootstrap

2026-10-20 的真实权重原型验证包含 25 项测试、4 个并发请求、64 token 输出。cached/recompute 两种模式的 128 次选定 token 概率观测通过，最大绝对差约 1.59e-5。该轮用于检查运行时与独立验收，不是构造器性能比较。

## 实际 Agent 构造迭代

2026-10-30 的生成调用使用 Codex CLI 与账户可用的 gpt-5.6-sol，固定起点为本项目初版逐请求运行时。生成器只修改 `engine.py` 和 `server.py`，增加批量 decode 及调度；评测代码保持固定。

调用耗时 77.69 秒；事件报告 input tokens 258104，其中 cached input 226560，output 6359。这些是调用用量记录，没有据此推算未知费用。

候选先通过参考模型、缓存、并发流式和原有负例验收，再执行固定开发负载：

- 模型：Qwen2.5-0.5B-Instruct，float32。
- 硬件与负载：单 A40，4 个并发请求，每请求 16 个输出 token。
- 对照：本项目初版逐请求运行时 A，与 Agent 生成的批量解码候选 B。
- 顺序与重复：A-B-B-A；每次预热后测量 3 次。

| 实现 | 第一次中位吞吐 | 第二次中位吞吐 |
| --- | ---: | ---: |
| 本项目初版逐请求运行时 | 63.92 token/s | 64.16 token/s |
| Agent 生成的批量解码候选 | 150.82 token/s | 148.80 token/s |

配对比值的几何平均约为 **2.34 倍**。原始样本和各次中位数见 [运行记录](../results/campaign-20261030.json)。

另外使用 3、17、31 token 的不同前缀，连续执行 4 步批量解码，共进行 12 次完整词表 logits 与独立 Hugging Face 参考比较。最大绝对误差为 5.34e-5，已测 argmax 一致，原始缓存状态保持不变。

## 结果适用范围

该结果表明，真实 Agent 修改可以被独立执行验证，并在上述小负载中改进自身初版。每个实现只有两次独立执行；它不构成广泛任务、多生成种子或生产 SLO 结论，也不是与 vLLM、SGLang、VibeServe 的性能对照。批量 decode 本身是成熟优化，本项目关注完整构造过程与执行反馈。

独立 8B 实验使用另一套引擎、精度与负载。联合存储/会话候选在三次开发重复中的总耗时中位数降低约 0.815%，存在波动，尚不足以支持稳健优势。恢复组件的局部加速也不能替代完整请求收益。完整样本、组件对照和限制见 [8B 结果](../experiments/qwen3_8b/README.md#结果与未解决的问题)。

## 重放历史候选

实验中生成的两个原始文件位于 [v0.2.0 的 examples/generated/batched/](https://github.com/yanchaomei/inference-forge/tree/v0.2.0/examples/generated/batched)。从独立的 v0.2.0 checkout 复现，使用该版本的 [运行指南](https://github.com/yanchaomei/inference-forge/blob/v0.2.0/docs/QUICKSTART.md) 和任务配置。需要通过原评测流程验收候选时，在这份独立 checkout 中将两个生成文件分别复制为 `forge/engine.py`、`forge/server.py`，再运行固定任务。

当前版本的目录职责已调整：

| 位置 | 当前内容 |
| --- | --- |
| `forge/backends/qwen2/engine.py` | 包内生成数值后端 |
| `forge/serving.py`、`forge/protocol.py` | 当前批处理服务与 SDK 协议 |
| `examples/generated/batched/` | 导入及启动兼容入口 |
| `forge/engine.py`、`forge/server.py` | 构造流程的原始种子 |

当前的兼容包装文件不应作为原始候选覆盖到种子目录。历史性能记录仍绑定原测量版本；新 CLI 和聊天接口的验证应单独记录。

公开数据去除了内部机器地址和个人目录，保留模型、精度、输出长度、源码散列与生成用量。模型权重从官方来源单独准备。
