import os

from dotenv import load_dotenv

from huggingface_hub import snapshot_download
from datasets import load_dataset

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    1. Download model.
"""

snapshot_download(
    repo_id="google-t5/t5-3b",
    local_dir=r"/home/dateng/model/huggingface/google-t5/t5-3b",
    local_dir_use_symlinks=False,
    ignore_patterns=["pytorch*", "tf_*"],
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

"""
    2. Load dataset.
"""

# dataset = load_dataset(r"/home/dateng/dataset/huggingface/yelp_review_full")
