from dotenv import load_dotenv

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import AutoModelForSpeechSeq2Seq, AutoFeatureExtractor, AutoTokenizer, AutoProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BitsAndBytesConfig
from transformers import AutomaticSpeechRecognitionPipeline

from peft import prepare_model_for_kbit_training, get_peft_model
from peft import PeftConfig, PeftModel
from peft import LoraConfig, LoraModel

from datasets import load_dataset, DatasetDict, Audio

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    Step 1: Basic configuration
"""

# repo_id
dataset_repo_id = "mozilla-foundation/common_voice_11_0"
# Dataset dir
dataset_dir = r"/home/dateng/dataset/huggingface/mozilla-foundation/common_voice_11_0"
# Model dir
model_dir = r"/home/dateng/model/huggingface/openai/whisper-large-v3"
# Output dir
output_dir = r"/home/dateng/model/huggingface/openai/peft/whisper-large-v3-lora"

"""
    Step 2: Load various processors
"""

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, language="Chinese (China)", task="transcribe")

# Load processor
processor = AutoProcessor.from_pretrained(model_dir, language="Chinese (China)", task="transcribe")

"""
    Step 3: Dataset preprocessing
"""

common_voice = DatasetDict()

# Load train dataset
common_voice["train"] = load_dataset(
    dataset_repo_id,
    name="zh-CN",
    split="train",
    trust_remote_code=True,
    cache_dir=dataset_dir
)

# Load validation dataset
common_voice["validation"] = load_dataset(
    dataset_repo_id,
    name="zh-CN",
    split="validation",
    trust_remote_code=True,
    cache_dir=dataset_dir
)

# Remove unnecessary fields
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)

# Down sampling
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


# Batch data processing func
def prepare_dataset(batch):
    audio = batch["audio"]
    # Down sampling to 16kHz
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# Apply preprocessing function to dataset
tokenized_common_voice = common_voice.map(prepare_dataset, num_proc=8)


# 定义一个针对语音到文本任务的数据整理器类
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # 处理器结合了特征提取器和分词器

    # 整理器函数，将特征列表处理成一个批次
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 从特征列表中提取输入特征，并填充以使它们具有相同的形状
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 从特征列表中提取标签特征（文本令牌），并进行填充
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 使用-100替换标签中的填充区域，-100通常用于在损失计算中忽略填充令牌
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果批次中的所有序列都以句子开始令牌开头，则移除它
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # 将处理过的标签添加到批次中
        batch["labels"] = labels

        return batch  # 返回最终的批次，准备好进行训练或评估


# Initialize data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

"""
    Step 4: Load model and peft config
"""

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_dir,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto"
)
# 这通常用于指定在解码（生成文本）过程中必须使用的特定token的ID，设置为None表示没有这样的强制要求
whisper_model.config.forced_decoder_ids = None
# 这用于指定在生成过程中应被抑制（不生成）的token的列表，设置为空列表表示没有要抑制的token
whisper_model.config.suppress_tokens = []
# 将所有非 int8 模块转换为全精度（fp32）以保证稳定性
# 为输入嵌入层添加一个 forward_hook，以启用输入隐藏状态的梯度计算
# 启用梯度检查点以实现更高效的内存训练
whisper_model = prepare_model_for_kbit_training(whisper_model)

# PEFT config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,  # MatMul(B,A) * (lora_alpha / r)
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

# PEFT model
peft_model = get_peft_model(whisper_model, lora_config)

peft_model.use_cache = False

"""
    Step 5: Train and save model
"""

# Configure training hyperparameters
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-3,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    # per_device_eval_batch_size=64,
    # evaluation_strategy="epoch",
    warmup_steps=50,
    # fp16=True,
    generation_max_length=128,
    logging_steps=100,
    remove_unused_columns=False,
    label_names=["labels"]
)

# Create Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=peft_model,
    train_dataset=tokenized_common_voice["train"],
    # eval_dataset=tokenized_common_voice["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor
)

# # Start training
# trainer.train()

# # Save model
# trainer.save_model(output_dir)

"""
    Step 6: Use the PEFT model
"""

peft_config = PeftConfig.from_pretrained(output_dir)

base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto"
)

peft_model = PeftModel.from_pretrained(base_model, output_dir)

peft_tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path,
    language="Chinese (China)",
    task="transcribe"
)

peft_processor = AutoProcessor.from_pretrained(
    peft_config.base_model_name_or_path,
    language="Chinese (China)",
    task="transcribe"
)

forced_decoder_ids = peft_processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
peft_model.generation_config.forced_decoder_ids = forced_decoder_ids

pipeline = AutomaticSpeechRecognitionPipeline(
    model=peft_model,
    tokenizer=peft_tokenizer,
    feature_extractor=peft_processor.feature_extractor
)

with torch.cuda.amp.autocast():
    text = pipeline(r"/home/dateng/dataset/peft/whisper-large-v3-lora/test_zh.flac", max_new_tokens=255)["text"]
