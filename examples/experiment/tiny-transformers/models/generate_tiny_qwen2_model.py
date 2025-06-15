from transformers import RecurrentGemmaModel, RecurrentGemmaConfig
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers import Qwen2Tokenizer
import os
import json


if __name__ == '__main__':
    config_path = "../configs/tiny_qwen2_config.json"
    output_dir = "../saved_models/tiny-qwen2"

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    configuration = Qwen2Config(**config_dict)

    # 初始化一个基于该配置的小模型（随机权重）
    model = Qwen2ForCausalLM(configuration)

    configuration = model.config
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model saved to {output_dir}")

    tokenizer.save_pretrained(output_dir)
    print(f"tokenizer saved to {output_dir}")

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")

    # tokenizer("Hello world")["input_ids"]
    prompt_1 = "Give me a short introduction to large language model."
    messages_1 = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt_1}
    ]

    messages_2 = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "你是谁"}
    ]

    text_1 = tokenizer.apply_chat_template(
        messages_1,
        tokenize=False,
        add_generation_prompt=True
    )

    text_2 = tokenizer.apply_chat_template(
        messages_2,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text_1, text_2], return_tensors="pt", padding=True)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # generate_ids = model.generate(inputs.input_ids, max_length=30)
    # tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(configuration)
    print(response)
