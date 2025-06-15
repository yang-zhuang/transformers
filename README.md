任务：通过修改原始模型的参数配置（如层数、隐藏层大小、注意力头数等），使其模型体积显著缩小，达到 10MB+ 的目标。
## ✅ 总体思路

你将使用 `transformers` 提供的 API 来：

1. **定义一个更小的模型配置（Config）**
2. **基于该配置随机初始化一个模型**
3. **保存这个小模型**
4. （可选）加载并验证模型结构和大小

这样你就能得到一个**参数量远小于原始大模型的小模型**。

---

## 🧱 一、项目结构建议（更新）

由于你现在主要是通过修改配置来生成小模型，可以简化之前的目录结构为如下形式：

```
tiny-transformers/
│
├── configs/                      # 存放不同模型配置文件
│   ├── tiny_bert_config.json     # 自定义的小型BERT配置
│   └── tiny_gpt2_config.json
│
├── models/                       # 用于生成和保存模型的脚本
│   ├── generate_tiny_model.py    # 主脚本：读取配置、生成并保存模型
│
├── utils/                        # 工具函数
│   └── model_size_utils.py       # 模型大小统计工具
│
├── saved_models/                 # 保存的小模型
│   ├── tiny-bert/
│   └── tiny-gpt2/
│
├── README.md
└── requirements.txt
```

---

## 📦 二、示例代码实现

### 1. 定义一个小型 BERT 配置（`configs/tiny_bert_config.json`）

你可以手动创建一个比原始 `bert-base-uncased` 小很多的配置文件：

```json
{
  "model_type": "bert",
  "vocab_size": 30522,
  "hidden_size": 128,           // 原始是 768
  "num_hidden_layers": 4,       // 原始是 12
  "num_attention_heads": 4,     // 原始是 12
  "intermediate_size": 512,     // 原始是 3072
  "max_position_embeddings": 512,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "use_cache": true
}
```

---

### 2. 生成并保存小模型的主脚本（`models/generate_tiny_model.py`）

```python
from transformers import BertConfig, BertForMaskedLM
import os
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return BertConfig(**config_dict)

def main():
    config_path = "../configs/tiny_bert_config.json"
    output_dir = "../saved_models/tiny-bert"

    config = load_config(config_path)
    print("Model Config:\n", config)

    # 初始化一个基于该配置的小模型（随机权重）
    model = BertForMaskedLM(config)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
```

---

### 3. 查看模型大小的工具（`utils/model_size_utils.py`）

```python
import os
import torch

def get_model_size(model):
    """
    获取模型占用磁盘空间大小（MB）
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return round(size_mb, 2)
```

在主脚本中加入调用即可显示保存后的模型文件大小。

---

## 📈 三、预期效果对比

| 模型 | 层数 | 隐藏维度 | 参数量 | 文件大小 |
|------|------|-----------|--------|----------|
| `bert-base` | 12 | 768 | ~110M | ~400MB |
| **tiny-bert** | 4 | 128 | ~1.5M | **~6MB** |

✅ 这样你就可以轻松得到一个 **小于 10MB 的小模型**！

---

## 🧩 四、后续扩展方向（可选）

虽然你现在只是修改配置，但未来你可以尝试：

- 使用 `AutoModel.from_config()` 支持多种模型（GPT2、RoBERTa 等）
- 添加训练脚本进行微调
- 使用 `safetensors` 格式保存，进一步压缩模型体积
- 编写模型推理脚本
- 构建 Web Demo（如 Gradio）

---

## 📝 五、README 示例内容

```markdown
# Tiny Transformers

该项目旨在通过调整 HuggingFace Transformers 中的模型配置，生成体积小于 10MB 的小模型，适合资源受限环境或教学演示。

## 当前成果

- ✅ 成功生成一个 6MB 左右的 Tiny-BERT 模型
- ✅ 可扩展支持 GPT2、DistilBERT 等架构

## 如何运行

```bash
cd models
python generate_tiny_model.py
```

生成的模型位于 `saved_models/` 目录下。
```

---

## ✅ 六、总结

你现在要做的就是：

1. 创建上述项目结构
2. 编写配置文件（JSON）
3. 编写生成模型的 Python 脚本
4. 保存并查看模型大小
5. 多个模型可以分别写成多个 `.py` 文件（如 `generate_tiny_gpt2.py`）

---

如果你希望我为你生成完整的项目模板（包括所有文件结构 + 示例代码），请告诉我 👍 我可以一次性打包发给你，方便你直接开始开发。

祝你早日完成自己的“小模型工程”！需要任何具体帮助都可以继续问我 😊