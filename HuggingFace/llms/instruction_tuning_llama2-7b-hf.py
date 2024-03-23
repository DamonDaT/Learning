from dotenv import load_dotenv

import torch

from HuggingFace.utils.llama_patch import replace_attn_with_flash_attn

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments

from peft import prepare_model_for_kbit_training, get_peft_model
from peft import TaskType, LoraConfig

from trl import SFTTrainer

from datasets import load_dataset

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    Step 1: Basic configuration
"""

# Dataset dir
dataset_dir = r"/home/dateng/dataset/huggingface/databricks/databricks-dolly-15k"
# Model dir
model_dir = r"/home/dateng/model/huggingface/meta-llama/Llama-2-7b-hf"
# Output dir
output_dir = r"/home/dateng/model/huggingface/meta-llama/Llama-2-7b-hf/instruction-tuning/databricks-dolly-15k"

"""
    Step 2: Load various processors
"""

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

"""
    Step 3: Dataset preprocessing
"""

# Load dataset
dataset = load_dataset(dataset_dir)


# Alpaca-Style from https://github.com/tatsu-lab/stanford_alpaca#data-release
def format_instruction(sample_data):
    """
    Formats the given data into a structured instruction format.

    Parameters:
        sample_data (dict): A dictionary containing 'response' and 'instruction' keys.

    Returns:
        str: A formatted string containing the instruction, input, and response.
    """
    # Check if required keys exist in the sample_data
    if 'response' not in sample_data or 'instruction' not in sample_data:
        # Handle the error or return a default message
        return "Error: 'response' or 'instruction' key missing in the input data."

    return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM. 

### Input:
{sample_data['response']}

### Response:
{sample_data['instruction']}
"""


"""
    Step 4: Load model and config
"""

# Use flash attention
replace_attn_with_flash_attn()

# Quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Quantization model
quantization_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto"
)
# Experimental feature, parallelization level during pretraining
quantization_model.config.pretraining_tp = 1
# Pre-operation for training
quantization_model = prepare_model_for_kbit_training(quantization_model)

# PEFT config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
)

# PEFT model
qlora_model = get_peft_model(quantization_model, lora_config)
qlora_model.print_trainable_parameters()

"""
    Step 5: Train and save model
"""

# Configure training hyperparameters
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    bf16=True,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    logging_steps=10,
    save_strategy="epoch",
    save_steps=50,
    output_dir=output_dir,
)

# Create trainer
trainer = SFTTrainer(
    model=qlora_model,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=training_args,
)

# # Start training
# trainer.train()
#
# # Save model
# trainer.save_model(output_dir)

"""
    Step 6: Reasoning based on fine-tuned model
"""


