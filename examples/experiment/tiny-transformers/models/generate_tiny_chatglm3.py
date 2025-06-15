# from transformers import RecurrentGemmaModel, RecurrentGemmaConfig
# from transformers import Qwen2ForCausalLM, Qwen2Config
# from transformers import Qwen2Tokenizer
import os
import json
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from transformers import QWenLMHeadModel
from transformers import QWenConfig
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import ChatGLMForConditionalGeneration, ChatGLMConfig

if __name__ == '__main__':
    config_path = "../configs/tiny_chatglm3_config.json"
    output_dir = "../saved_models/tiny-chatglm3"

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    configuration = ChatGLMConfig(**config_dict)

    # 初始化一个基于该配置的小模型（随机权重）
    model = ChatGLMForConditionalGeneration(configuration)

    configuration = model.config
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model saved to {output_dir}")

    tokenizer.save_pretrained(output_dir)
    print(f"tokenizer saved to {output_dir}")

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")

    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
