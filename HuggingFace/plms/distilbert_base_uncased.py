from dotenv import load_dotenv

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Dataset dir
dataset_dir = r"/home/dateng/dataset/huggingface/squad_v2"
# Model dir
model_dir = r"/home/dateng/model/huggingface/distilbert/distilbert-base-uncased"
# Output dir
output_dir = r"/home/dateng/model/huggingface/distilbert/fine-tune/distilbert-base-uncased-finetune-suqad-v2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=5)
# Load dataset
dataset = load_dataset(dataset_dir)
