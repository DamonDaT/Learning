from dotenv import load_dotenv

import torch

from transformers import OPTForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

from datasets import load_dataset

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    Step 1: Basic configuration
"""

# Dataset dir
dataset_dir = r"/home/dateng/dataset/huggingface/Abirate/english_quotes"
# Model dir
model_dir = r"/home/dateng/model/huggingface/facebook/opt-6.7b"
# Output dir
output_dir = r"/home/dateng/model/huggingface/facebook/peft/opt-6.7b-lora"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

"""
    Step 2: Dataset preprocessing
"""

# Load dataset
dataset = load_dataset(dataset_dir)
# Apply preprocessing function to dataset
tokenized_datasets = dataset.map(lambda samples: tokenizer(samples["quote"]), batched=True)
# 数据收集器，用于处理语言模型的数据，这里设置为不使用掩码语言模型(MLM)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

"""
    Step 3: Model preprocessing
"""

# Quantization model
lora_model = OPTForCausalLM.from_pretrained(model_dir, load_in_8bit=True)

# 将所有非 int8 模块转换为全精度（fp32）以保证稳定性
# 为输入嵌入层添加一个 forward_hook，以启用输入隐藏状态的梯度计算
# 启用梯度检查点以实现更高效的内存训练
lora_model = prepare_model_for_kbit_training(lora_model)

# PEFT config
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,  # MatMul(B,A) * (lora_alpha / r)
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
    lora_dropout=0.05,
    bias="none"
)

# Reload model
lora_model = get_peft_model(lora_model, lora_config)

lora_model.use_cache = False

# # Calculate training parameters
# lora_model.print_trainable_parameters()

"""
    Step 4: Train
"""

# Configure training hyperparameters
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    fp16=True,  # 混合精度
    logging_steps=20,
    num_train_epochs=5
)

# Create Trainer
trainer = Trainer(
    model=lora_model,
    train_dataset=tokenized_datasets["train"],
    args=training_args,
    data_collator=data_collator,
)

# Start training
trainer.train()

"""
    Step 5: Save peft model
"""

lora_model.save_pretrained(output_dir)

"""
    Step 6: Load and run peft model
"""

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

# Load model
lora_model = OPTForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

# Tokenization
inputs = tokenizer("Two things are infinite: ", return_tensors="pt").to("cuda")

# Generate
outputs = tokenizer.decode(lora_model.generate(**inputs, max_new_tokens=64)[0], skip_special_tokens=True)
