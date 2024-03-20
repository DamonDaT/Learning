from dotenv import load_dotenv

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model dir
model_dir = r"/home/dateng/model/huggingface/facebook/opt-6.7b"
# Output dir
output_dir = r"/home/dateng/model/huggingface/facebook/quantization/opt-6.7b-bnb4"

# Quantization config
bnb4_qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# # Quantization model
# bnb4_model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     quantization_config=bnb4_qlora_config,
#     device_map="auto"
# )

# # Save model
# bnb4_model.save_pretrained(output_dir)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load model
bnb4_model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

# Tokenization
inputs = tokenizer("Merry Christmas! I'm glad to", return_tensors="pt").to("cuda")

# Generate
outputs = tokenizer.decode(bnb4_model.generate(**inputs, max_new_tokens=64)[0], skip_special_tokens=True)
