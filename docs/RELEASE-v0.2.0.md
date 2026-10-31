# v0.2.0 — 可运行源码、8B实验与参赛材料

该版本将可直接运行的0.5B推理服务、独立8B存储与推理实验，以及作品介绍和演示视频统一到一个公开入口。

- [完整源码与运行指南](https://github.com/yanchaomei/inference-forge/blob/v0.2.0/docs/QUICKSTART.md)：安装、固定模型版本、服务启动、普通请求、中文SSE和远程端口转发。
- [作品介绍PDF](https://yanchaomei.github.io/inference-forge/project-introduction.pdf)：项目背景、架构、场景、构造反馈及8B联合实验。
- [演示播放页](https://yanchaomei.github.io/inference-forge/)与[视频直链](https://yanchaomei.github.io/inference-forge/demo.mp4)。
- [8B实验入口](https://github.com/yanchaomei/inference-forge/blob/v0.2.0/experiments/qwen3_8b/README.md)：主机KV恢复、配套会话消费者、真实构造和四格检查。

下载完整源码请使用本Release自动提供的Source code ZIP或按tag克隆。模型权重单独下载；运行已有引擎不需要Coding Agent账号。

发布前验证：25项CPU检查；真实A40上的HTTP/SSE、四请求并发和输入限制；12次完整词表参考比较。详细结果见[验证记录](https://github.com/yanchaomei/inference-forge/blob/v0.2.0/results/release-v0.2.0-validation.json)。

视频保留2026-10-30的90秒0.5B字幕回放，等待压缩、无配音；不是8B实时演示。8B联合收益仍不充分，本版不宣称成熟框架性能优势。材料已公开发布，比赛门户提交需另行完成。
