# Qwen3-8B 配套存储与会话消费者

真实构造Agent根据反馈生成了`consumer.py`与`kv_store.py`。前者维护GPU驻留会话、待消费的末token与LRU驱逐；后者保留版本化主机前缀并接受增量追加。主机同步在实际驱逐时发生，避免每轮主动备份。

消费者会在可选接口不完整时使用通用put/get/delete。增量接口要求受信消费者保持已提交前缀不变，版本句柄并不验证任意张量内容是否被修改。当前冻结engine以新张量追加KV，满足此条件。

`SessionRuntime(engine, store, gpu_budget_bytes)`提供`turn/close/stats`；`turn`返回固定长度greedy输出，支持独立会话续写。无HTTP服务、取消和并发线程控制。

全部四格组合在开发trace上保持完整输出并遵守预算，但联合中位提升仅0.815%，尚无稳健优势结论。数据、源码身份和可执行复现入口见 [8B实验说明](../../experiments/qwen3_8b/README.md)。
