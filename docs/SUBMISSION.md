# OA2026 参赛提交入口

项目：**InferenceForge — 面向模型、硬件与场景的专用推理引擎构造工具链**。

适用方向：开源基础软件与解决方案。代码采用 MIT 许可证，模型权重按发布者许可单独下载。

| 提交项 | 可直接使用的链接 |
|---|---|
| 完整代码仓库 | https://github.com/yanchaomei/inference-forge |
| 作品介绍文档（PDF） | https://yanchaomei.github.io/inference-forge/project-introduction.pdf |
| 演示视频播放页 | https://yanchaomei.github.io/inference-forge/ |
| 视频文件直链 | https://yanchaomei.github.io/inference-forge/demo.mp4 |
| 固定运行版本 | https://github.com/yanchaomei/inference-forge/releases/tag/v0.2.0 |

## 推荐查看顺序

1. 阅读34页作品介绍：项目背景、场景、技术架构、生成与执行反馈、8B存储和推理联合实验。
2. 观看90秒演示：任务定义、构造、固定GPU检查、生成产物和结果记录。
3. 按[快速运行](QUICKSTART.md)启动公开0.5B引擎，检查服务、普通请求和SSE流式输出。
4. 按[8B实验指南](../experiments/qwen3_8b/README.md)检查主机KV恢复与多会话消费者，或构造新候选。

## 视频与文档版本

当前视频是2026-10-30真实0.5B运行记录的90秒字幕回放，等待已压缩，无配音；不是8B实时服务演示。34页详细PDF覆盖场景、架构、构造反馈、运行时、KV存储、会话管理、实验和复现，运行代码依据v0.2.0（`03b8a0a`）。其中4张生成式封面/机制图与4张原始数据统计图分别记录来源。

[可编辑正文](technical-report.md)与[完整文档源文件](https://github.com/yanchaomei/inference-forge/releases/download/submission-20261031/InferenceForge-report-source.zip)同时提供，包含排版程序、图片和绘图数据。

0.5B演示的约2.34倍仅是相对本项目初版的开发对照。8B联合中位改善约0.815%，目前不足以证明稳健端到端优势；未宣称优于vLLM/SGLang。核心交付是可运行源码、构造工具链、独立验收与可回读证据。

以上是公开交付链接，不能代替比赛门户的正式提交回执。
