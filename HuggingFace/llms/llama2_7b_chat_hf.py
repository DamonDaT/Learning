from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModel

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model dir
model_dir = r"/home/dateng/model/huggingface/meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load Model
model = AutoModel.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)

# Prompt
prompt = "如何理解马斯克的时间拳法"

# Tokenization
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
