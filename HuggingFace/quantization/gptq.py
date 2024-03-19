from dotenv import load_dotenv

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model dir
model_dir = r"/home/dateng/model/huggingface/facebook/opt-6.7b"
# Output dir
output_dir = r"/home/dateng/model/huggingface/facebook/quantization/opt-6.7b-gptq"

# Quantization config
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="wikitext2",
    desc_act=False
)

# # Quantization model
# gptq_model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     quantization_config=gptq_config,
#     device_map="auto"
# )

# # Save model
# gptq_model.save_pretrained(output_dir)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load model
gptq_model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

# Tokenization
inputs = tokenizer("Merry Christmas! I'm glad to", return_tensors="pt").to("cuda")

# Generate
outputs = tokenizer.decode(gptq_model.generate(**inputs, max_new_tokens=64)[0], skip_special_tokens=True)
