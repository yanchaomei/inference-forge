# 直接运行与演示

先运行已生成的Qwen2.5-0.5B引擎，再按需使用构造器或8B实验。已有引擎运行时不需要Coding Agent、API密钥或远程模型服务。

## 1. 准备环境

Linux、Python 3.10以上、NVIDIA GPU及可用驱动。已验证单A40；以下0.5B示例使用float32，总上下文最多512 token，输出1–64 token。8B依赖及运行范围见[独立指南](../experiments/qwen3_8b/README.md)。

```bash
git clone --branch v0.2.0 https://github.com/yanchaomei/inference-forge.git
cd inference-forge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference,test]'
nvidia-smi
```

后续命令在仓库根目录执行。选择空闲GPU；下例选择编号0：

```bash
export CUDA_VISIBLE_DEVICES=0
export FORGE_MODEL_PATH="$PWD/cache/models/Qwen2.5-0.5B-Instruct"
```

## 2. 下载模型

权重从Qwen官方仓库单独下载，固定本示例revision：

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    local_dir=os.environ["FORGE_MODEL_PATH"],
)
PY
```

已有模型时直接将`FORGE_MODEL_PATH`指向本地目录。模型与tokenizer下载完成后，推理服务离线读取权重。固定验收会核对权重身份；更换版本需另行记录。

## 3. 启动已生成的服务

```bash
python -m examples.generated.batched.server \
  --model "$FORGE_MODEL_PATH" --dtype float32 --port 8000 \
  --event-log runs/demo-events.jsonl
```

看到`"event": "ready"`后，在另一终端执行：

```bash
curl --fail http://127.0.0.1:8000/health

curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16}'

curl --fail -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"中国科学技术大学位于","max_tokens":16,"stream":true}'
```

健康检查返回`ready: true`；普通请求返回`text`、`token_ids`、`usage`；流式请求返回逐token事件和最终`data: [DONE]`。这是项目自身的completions协议，不是聊天界面，也不承诺完整OpenAI API兼容。固定长度greedy输出的结束原因通常为`length`。

服务监听`127.0.0.1`。使用远程GPU时，在本地另一终端建立转发，再在本地访问同一地址：

```bash
ssh -L 8000:127.0.0.1:8000 user@gpu-host
```

结束演示时，在服务终端按`Ctrl+C`。请求接收、调度与完成事件保存在`runs/demo-events.jsonl`。

## 4. 检查代码与构造新候选

无GPU的环境可仅安装`.[test]`，检查构造与任务接口：

```bash
python -m pytest tests/test_campaign.py tests/test_task.py \
  tests/test_evaluate.py tests/test_replay.py tests/test_processes.py \
  tests/test_8b_export.py -q
```

生成新引擎使用已登录的Codex CLI及独立GPU评测环境，完整命令见[README构造流程](../README.md#生成和优化新引擎)。8B的KV保存/恢复、多会话组合和构造入口见[8B实验指南](../experiments/qwen3_8b/README.md)。

## 常见问题

| 现象 | 处理 |
|---|---|
| CUDA不可用 | 检查`nvidia-smi`、GPU选择及所装PyTorch是否支持CUDA |
| 找不到模型或tokenizer | 确认下载完整，模型路径指向包含`config.json`的目录 |
| 端口已被占用 | 更换启动参数`--port`，并同步修改请求地址 |
| 请求400 | 输入与输出总长度需≤512；`max_tokens`需为1–64的整数 |
| 没有Codex登录 | 可以直接运行已有服务；只有构造新候选时需要生成器登录 |

演示视频及作品说明见[提交入口](SUBMISSION.md)。视频展示0.5B构造实例；8B实验尚无完整HTTP服务与成熟框架性能优势。

2026-10-31已在单A40、Torch2.11环境重新验证上述服务入口：普通及流式输出一致、四请求并发正常、无效长度返回400，12次完整词表参考比较通过。原始公开回执见[发布验证记录](../results/release-v0.2.0-validation.json)。
