from dotenv import load_dotenv

import numpy as np
import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Dataset dir
dataset_dir = r"/home/dateng/dataset/huggingface/yelp_review_full"
# Model dir
model_dir = r"/home/dateng/model/huggingface/google-bert/bert-base-cased"
# Output dir
output_dir = r"/home/dateng/model/huggingface/google-bert/fine-tune/bert-base-cased-finetune-yelp"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=5)
# Load dataset
dataset = load_dataset(dataset_dir)


# Preprocess dataset
def tokenize_func(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Apply preprocessing function to dataset
tokenized_datasets = dataset.map(tokenize_func, batched=True)

# Data sampling
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Configure hyperparameters
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    logging_steps=1000
)

# Evaluation metric
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Save model and state
trainer.save_model(output_dir)
trainer.save_state()
