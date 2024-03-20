from dotenv import load_dotenv

import torch
from typing import List, Dict, Optional

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer

from peft import prepare_model_for_kbit_training, get_peft_model
from peft import TaskType, LoraConfig
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from datasets import load_dataset

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    Step 1: Basic configuration
"""

# Dataset dir
dataset_dir = r"/home/dateng/dataset/huggingface/HasturOfficial/adgen"
# Model dir
model_dir = r"/home/dateng/model/huggingface/THUDM/chatglm3-6b"
# Output dir
output_dir = r"/home/dateng/model/huggingface/THUDM/peft/chatglm3-6b-qlora"

# 定义全局变量和参数
eval_data_path = None  # 验证数据路径，如果没有则设置为None
seed = 8  # 随机种子
max_input_length = 512  # 输入的最大长度
max_output_length = 1536  # 输出的最大长度
lora_rank = 4  # LoRA秩
lora_alpha = 32  # LoRA alpha值
lora_dropout = 0.05  # LoRA Dropout率
resume_from_checkpoint = None  # 如果从checkpoint恢复训练，指定路径
prompt_text = ''  # 所有数据前的指令文本
compute_dtype = 'fp32'  # 计算数据类型（fp32, fp16, bf16）

"""
    Step 2: Load various processors
"""

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, revision='b098244')

"""
    Step 3: Dataset preprocessing
"""

# Load dataset
dataset = load_dataset(dataset_dir)


# Batch data processing func
def tokenize_func(example, tokenizer, ignore_label_id=-100):
    """
    对单个数据样本进行tokenize处理。

    Args:
        example (dict): 包含'content'和'summary'键的字典，代表训练数据的一个样本。
        tokenizer (transformers.PreTrainedTokenizer): 用于tokenize文本的tokenizer。
        ignore_label_id (int, optional): 在label中用于填充的忽略ID，默认为-100。

    Return:
        dict: 包含'tokenized_input_ids'和'labels'的字典，用于模型训练。
    """

    # 构建问题文本
    question = prompt_text + example['content']
    if example.get('input', None) and example['input'].strip():
        question += f'\n{example["input"]}'

    # 构建答案文本
    answer = example['summary']

    # 对问题和答案文本进行tokenize处理
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    # 如果tokenize后的长度超过最大长度限制，则进行截断
    if len(q_ids) > max_input_length - 2:  # 保留空间给gmask和bos标记
        q_ids = q_ids[:max_input_length - 2]
    if len(a_ids) > max_output_length - 1:  # 保留空间给eos标记
        a_ids = a_ids[:max_output_length - 1]

    # 构建模型的输入格式
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2  # 加上gmask和bos标记

    # 构建标签，对于问题部分的输入使用ignore_label_id进行填充
    labels = [ignore_label_id] * question_length + input_ids[question_length:]

    return {'input_ids': input_ids, 'labels': labels}


# Apply preprocessing function to dataset
tokenized_train_dataset = dataset["train"].map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False,
    remove_columns=dataset["train"].column_names,
    num_proc=8
)

# Shuffle dataset
tokenized_train_dataset = tokenized_train_dataset.shuffle(seed=32)
# Rewrite the entire dataset to disk, removing the index mapping
tokenized_train_dataset = tokenized_train_dataset.flatten_indices()


# Data collator for ChatGLM
class DataCollatorForChatGLM:
    """
    用于处理批量数据的DataCollator，尤其是在使用 ChatGLM 模型时。

    该类负责将多个数据样本（tokenized input）合并为一个批量，并在必要时进行填充(padding)。

    属性:
        pad_token_id (int): 用于填充(padding)的token ID。
        max_length (int): 单个批量数据的最大长度限制。
        ignore_label_id (int): 在标签中用于填充的ID。
    """

    def __init__(self, pad_token_id: int, max_length: int = 2048, ignore_label_id: int = -100):
        """
        初始化DataCollator。

        参数:
        pad_token_id (int): 用于填充(padding)的token ID。
        max_length (int): 单个批量数据的最大长度限制。
        ignore_label_id (int): 在标签中用于填充的ID，默认为-100。
        """
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """
        处理批量数据。

        参数:
            batch_data (List[Dict[str, List]]): 包含多个样本的字典列表。

        返回:
            Dict[str, torch.Tensor]: 包含处理后的批量数据的字典。
        """

        # 计算批量中每个样本的长度
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)  # 找到最长的样本长度

        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d  # 计算需要填充的长度
            # 添加填充，并确保数据长度不超过最大长度限制
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[:self.max_length]
                label = label[:self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        # 将处理后的数据堆叠成一个tensor
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return {'input_ids': input_ids, 'labels': labels}


# Initialize data collator
data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)

"""
    Step 4: Load model and peft config
"""

# Quantization config
bnb4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model
chat_model = AutoModel.from_pretrained(
    model_dir,
    quantization_config=bnb4_config,
    device_map="auto",
    trust_remote_code=True,
    revision='b098244'
)

# 将所有非 int8 模块转换为全精度（fp32）以保证稳定性
# 为输入嵌入层添加一个 forward_hook，以启用输入隐藏状态的梯度计算
# 启用梯度检查点以实现更高效的内存训练
chat_model = prepare_model_for_kbit_training(chat_model)

# PEFT target modules
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["chatglm"]

# PEFT config
lora_config = LoraConfig(
    target_modules=target_modules,
    r=8,
    lora_alpha=32,  # MatMul(B,A) * (lora_alpha / r)
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)

# PEFT model
peft_model = get_peft_model(chat_model, lora_config)
peft_model.print_trainable_parameters()

"""
    Step 5: Train and save model
"""

# Configure training hyperparameters
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    # per_device_eval_batch_size=4,
    learning_rate=1e-3,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    # evaluation_strategy="steps",
    # eval_steps=500,
    optim="adamw_torch",
    fp16=True,
)

# Create Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save model
trainer.model.save_pretrained(output_dir)
