from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModel

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model dir
model_dir = r"/home/dateng/model/huggingface/THUDM/chatglm3-6b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Load Model
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model = model.eval()

# Use ChatGLM3-6B
response, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(tokenizer, "如何理解马斯克的时间拳法", history=history)
