import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
Args:
    repo_id: A user or an organization name and a repo name separated by a `/`.
    local_dir: If provided, the downloaded files will be placed under this directory.
    ignore_patterns: If provided, files matching any of the patterns are not downloaded.
    token: A token to be used for the download.
"""
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    local_dir=r"/home/dateng/model/huggingface/meta-llama/Llama-2-7b-chat-hf",
    ignore_patterns=["*.bin"],
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
