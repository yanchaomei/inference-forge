# 系统说明

InferenceForge 有两个入口：运行已有引擎，以及构造新候选。构造 Agent 参与代码生成与优化，推理服务独立执行。

## 运行已有引擎

`inference-forge serve MODEL_PATH` 读取本地模型，启动包内 Qwen2 引擎。当前支持 Qwen2.5-0.5B-Instruct、float32 和短上下文请求，具体范围见 [支持矩阵](SUPPORTED_MODELS.md)。

```text
CLI → HTTP 请求 → 协议校验 / 聊天模板 → 请求调度 → Qwen2 后端
                    ↑                              │
                    └──── JSON / SSE 输出 ─────────┘
```

协议层检查模型身份和生成参数，并把聊天消息通过 tokenizer 模板转换为 token。运行时维护独立请求的 KV 状态，调度 prefill 和批量 decode，再通过普通 JSON 或 SSE 返回结果。API 保留既有 token 字段，便于评测器直接核对生成序列。

| 文件 | 职责 |
| --- | --- |
| [forge/cli.py](../forge/cli.py) | 统一命令、环境检查、模型列表 |
| [forge/protocol.py](../forge/protocol.py) | 请求规范、聊天模板、SDK 响应格式 |
| [forge/serving.py](../forge/serving.py) | 批处理调度、HTTP/SSE 和事件记录 |
| [forge/backends/qwen2/engine.py](../forge/backends/qwen2/engine.py) | 权重读取、Qwen2 前向、KV 与批量 decode |

## 构造新候选

用户提供模型、设备、负载与约束。构造器保留可信评测工具，把种子运行时的 `engine.py` 和 `server.py` 交给 Coding Agent 修改，再将候选与固定评测代码组装为快照，发到远程 GPU 执行。

参考模型与候选分进程运行。参考产生 token、teacher-forced 概率与解码文本；候选接受缓存续写、并发流式请求和数值检查。通过正确性后，在固定负载下预热并重复测量。构造器根据结果选择下一轮起点，同时保留失败、日志和源码身份。

| 文件 | 职责 |
| --- | --- |
| [forge/campaign.py](../forge/campaign.py) | Agent 调用、候选装配、远程任务和反馈循环 |
| [forge/engine.py](../forge/engine.py) | 原始 Qwen2 构造种子的前向与 KV |
| [forge/server.py](../forge/server.py) | 种子服务的调度、流式接口和诊断观测 |
| [forge/evaluate.py](../forge/evaluate.py) | 独立参考、数值、文本与请求比较 |
| [forge/pilot.py](../forge/pilot.py) | 固定任务验收与记录 |
| [forge/replay.py](../forge/replay.py) | 保留失败签名的外部请求轨迹缩小 |

当前默认服务使用已生成后端；构造流程仍从原始种子出发。两者职责独立，修改默认服务不会自动改变构造起点。构造命令与开发检查见 [开发指南](DEVELOPMENT.md)。

## 生成产物与历史版本

0.5B 批量解码原始候选固定保存在 [v0.2.0 的 examples/generated/batched/](https://github.com/yanchaomei/inference-forge/tree/v0.2.0/examples/generated/batched)。当前同名目录是兼容包装，数值实现归入包内后端，服务与协议由各自模块维护。历史复现使用冻结版本，当前安装使用统一 CLI。

8B 的主机 KV 与会话消费者是独立实验，入口在 [experiments/qwen3_8b/](../experiments/qwen3_8b/README.md)，尚未接入当前服务。

新候选可按模型与负载选择算子、KV 布局和调度。可复用的是任务定义、执行工具和反馈流程；每个候选仍需对应的正确性与性能检查。下一步方向见 [特化路线](SPECIALIZATION.md) 和 [路线图](ROADMAP.md)。
