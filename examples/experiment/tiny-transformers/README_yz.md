如何设置配置？
可以参考已经开源且经过预训练、微调、RLHF完整阶段的模型，如：
HuggingFaceTB/SmolLM2-135M-Instruct
BAAI/Aquila-135M-Instruct

查看他们的config.json文件进行模仿，我们只需要修改一下num_hidden_layers即可，比如设置2层等等，这样就可以将模型大小降低很多

这只是模仿它们的模型，但是注意tokenizer我们需要直接用他们的