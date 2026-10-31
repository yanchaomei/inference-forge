# Qwen3-8B 状态恢复与多会话实验

这是可运行的研究实验区。代码来自真实构造与迭代，独立执行器检查每个实际输出；目前没有证明生成引擎优于vLLM/SGLang，也没有充分证明联合优化带来端到端优势。

## 产物

| 目录 | 内容 |
|---|---|
| `engines/qwen3_8b_kv/` | 先前生成并冻结的Qwen3引擎、两轮生成优化后的主机存储组件 |
| `engines/qwen3_8b_sessions/` | 实际生成的配套存储/消费者；版本化主机前缀、增量追加、驱逐时同步 |
| `baseline_consumer.py` | 手写通用LRU消费者，作为固定对照；不能计为生成产物 |
| `construct.py` | 本地构造入口；保存提示、源码身份、调用记录和用量，不自动宣称验收通过 |
| `session_matrix.py` | GPU执行入口；同一trace对比四种组件组合，并检查完整logits摘要和显存 |
| `host_roundtrip.py` | 单独复现主机存储的九组状态保持检查与恢复微基准 |

主要方法参考是 [BespokeOLAP](https://github.com/DataManagementLab/BespokeOLAP/tree/e5c5acd373a802d3f13f9be04387d79ca35a072a) 的存储/执行共同生成和真实反馈流程。本实验实现并非该项目的推理版复现。

## 已验证范围

Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`、BF16、TP1、单A40，最多4096总上下文。已测环境：Torch `2.10.0+cu128`、Transformers `4.57.6`、safetensors `0.8.0`。这是独立8B环境，仓库首页的Qwen2依赖组合用于原0.5B入口。

主机存储保存各层KV和下一步FP32 logits，使用锁页内存。`get`后的消费者在同一CUDA stream上有顺序保证；跨stream由调用者建立依赖。容量是已提交张量载荷限制，不是峰值进程RAM限制。

会话运行时保留最后一个尚未进入KV的输出token，下一轮与新增输入一起消费。可选增量接口依赖受信的冻结消费者不修改已提交前缀；版本句柄仅检查身份和位置，不证明任意张量内容不可变。通用put/get/delete仍保留无损与任意值语义。

本轮没有HTTP会话服务、NVMe、取消、多线程控制或自动故障恢复；失败的试验应保留日志并重新创建独立输出目录。

## 准备模型

在GPU服务器准备上述独立依赖环境，并按 [目标配置](../../targets/qwen3-8b-a40/target.json) 使用 `scripts/prepare_model.py` 准备权重。仓库不分发权重。评测器会逐文件核对 `model.json` 的官方身份。

```bash
export FORGE_MODEL_PATH=/path/to/Qwen3-8B-b968826
export FORGE_GPU_UUID=GPU-your-id
export FORGE_PYTHON=/path/to/8b-venv/bin/python
```

以下命令都在仓库根目录执行。确认指定GPU空闲后，在GPU服务器检查公开产物：

```bash
CUDA_VISIBLE_DEVICES="$FORGE_GPU_UUID" PYTHONPATH=. PYTHONNOUSERSITE=1 \
  "$FORGE_PYTHON" -m experiments.qwen3_8b.tests.check_candidate \
  --candidate engines/qwen3_8b_sessions

CUDA_VISIBLE_DEVICES="$FORGE_GPU_UUID" PYTHONPATH=. PYTHONNOUSERSITE=1 \
  "$FORGE_PYTHON" -m experiments.qwen3_8b.tests.check_retained_protocol \
  --candidate engines/qwen3_8b_sessions --output runs/protocol-check.json
```

结果文件不能已存在。生成与编辑在本地，模型推理和测试在GPU服务器执行。

## 真实构造新候选

在本地已登录Codex CLI的环境中运行，指定账户实际可用的模型。构造阶段不访问GPU或远程凭据。

```bash
python -m experiments.qwen3_8b.construct \
  --output runs/my-session-candidate --generator-model YOUR_AVAILABLE_MODEL \
  --clean-generator-config
```

该入口每次调用一个生成器，默认超时360秒；实际预算和调用结果写入manifest。它是新构造入口，不重放本报告的两轮构造，也不执行目标配置里另列的3600秒/3轮研究计划。

`--prepare-only`可仅准备接口和种子，不调用模型。成功调用但没有源码变化记作`no_code_change`；源码变化记作`generated_not_validated`，要经过下一步GPU评测。缺失必需文件会记录明确的拒绝状态。`runs/`默认不入Git，不能把个人构造日志直接公开。

## 四种组合的同条件评测

先将源码和候选同步到GPU服务器。在tmux中运行并保存日志，输出目录必须全新：

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 \
  "$FORGE_PYTHON" -m experiments.qwen3_8b.session_matrix \
  --model-path "$FORGE_MODEL_PATH" --gpu-uuid "$FORGE_GPU_UUID" \
  --candidate engines/qwen3_8b_sessions --output runs/session-evaluation --repeats 3
```

验证新构造产物时，把`--candidate`换成同步后的`runs/my-session-candidate/candidate`。评测包含官方权重核查、无换出参考、每种组合完整预热、三次顺序交错重复和全驻留对照。六个合成会话、24轮、每轮4输出token；完成一轮时的GPU会话状态预算768MiB，另记录瞬时峰值；主机已提交载荷预算4GiB。每个计时输出都比对完整logits摘要、token及logprob，计时包含观察开销。

这只是同一联合候选的组件交叉消融。它没有比较“两个独立构造器分别优化”的结果，也没有完成VibeSys/MetaInfer的同预算构造比较。

单独复现主机状态的九组完整检查：

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 \
  "$FORGE_PYTHON" -m experiments.qwen3_8b.host_roundtrip \
  --model-path "$FORGE_MODEL_PATH" --gpu-uuid "$FORGE_GPU_UUID" \
  --output runs/host-state-check --correctness-only
```

省略`--correctness-only`会在通过检查后执行恢复/重算微基准。GitHub CPU CI只检查源码/公开记录一致性、准备入口与语法；GPU验证单独记录，不能以CI绿灯代替。

## 结果与未解决的问题

| 组合 | 24轮总耗时中位数 |
|---|---:|
| 原存储＋原消费者 | 7160.08ms |
| 新存储＋原消费者 | 7166.64ms |
| 原存储＋新消费者 | 7287.06ms |
| 新存储＋新消费者 | 7101.71ms |

联合版本相对基线降低约0.815%，目前只有三次开发重复，且存在波动，**不足以支持稳健的联合优势**。更多长期会话、独立留出及成熟缓存系统对照仍需完成。

独立主机恢复实验中，第二版恢复速度为首版2.78–3.87倍，但恢复后16token总时间仅降低0.4%–4.6%；不能把组件加速当成整个框架加速。

可公开原始逐请求记录在 [多会话结果](../../results/qwen3-8b-sessions-20261031.json)，主机恢复摘要在 [主机恢复结果](../../results/qwen3-8b-host-state-20261031.json)，生成信息与源码身份在 [provenance.json](provenance.json)。记录不包含内部地址、GPU UUID、凭据或业务数据。
