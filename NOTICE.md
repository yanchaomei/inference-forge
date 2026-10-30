# 来源与依赖

本项目代码采用 MIT 许可证。模型权重不随仓库分发，下载与使用时遵循模型发布者的许可证。

- Qwen2.5-0.5B-Instruct：[官方模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。运行器实现 Qwen2 数学结构，参考及 tokenizer 来自 Transformers。
- [PyTorch](https://github.com/pytorch/pytorch)、[Transformers](https://github.com/huggingface/transformers)、[safetensors](https://github.com/huggingface/safetensors) 作为安装依赖，不在此复制其源码或变更许可证。
- 生成器通过用户自行安装和登录的 Codex CLI 调用。仓库不包含账户凭据，CLI 与模型服务受各自条款约束。
- 研究参考：[VibeServe](https://arxiv.org/abs/2605.06068)、[MetaInfer](https://arxiv.org/abs/2607.12875)、[Bespoke OLAP](https://arxiv.org/abs/2603.02001)。本版本的独立试验代码不等于上述系统的复现；后续引入其代码时按实际许可保留归属。

部分代码和文档由 Coding Agent 辅助生成，并经过远程真实模型验证和人工／独立代理审查。已验证范围与失败记录见运行报告。
# Design references

The model/hardware/workload-specific engine product direction is informed by
[Strata](https://github.com/Niko1221/Strata). Its design and reported performance
are referenced in `docs/SPECIALIZATION.md`; no Strata engine code or performance
result is included as an InferenceForge implementation or measurement.

