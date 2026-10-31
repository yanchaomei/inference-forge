# Qwen3-8B 主机状态恢复组件

`engine.py`是先前真实生成的完整Qwen3模型运行时，本轮冻结数学运算；`kv_store.py`是经过两轮生成与实测反馈的锁页主机存储组件。

接口：`KVStore(max_bytes).put/get/delete`及`used_bytes`。保存各层KV与下一步FP32 logits，通用写入保持任意张量值、形状和精度；读取返回独立副本。已提交载荷预算不等于主机峰值内存限制。面向CUDA/A40运行环境。

九组真实模型状态保存/恢复检查通过；对齐路径的后续完整logits及token逐位一致。该结论相对于固定引擎，不替代独立模型质量验收。

完整说明、依赖、构造及远程评测命令见 [8B实验入口](../../experiments/qwen3_8b/README.md)。
