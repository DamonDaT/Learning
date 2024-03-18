from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AwqConfig
from transformers.utils.quantization_config import AwqBackendPackingMethod, AWQLinearVersion

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model dir
model_dir = r"/home/dateng/model/huggingface/facebook/opt-6.7b"
# Output dir
output_dir = r"/home/dateng/model/huggingface/facebook/quantization/opt-6.7b-llm-awq"

# Quantization config
awq_config = AwqConfig(
    bits=4,
    group_size=128,
    zero_point=True,
    backend=AwqBackendPackingMethod.LLMAWQ,
    version=AWQLinearVersion.GEMV,
)

# Set the attribute `quantization_config` in model's config
config = AutoConfig.from_pretrained(model_dir)
config.quantization_config = awq_config

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
