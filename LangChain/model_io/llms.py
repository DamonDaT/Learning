import os
from typing import Any

from dotenv import load_dotenv

import torch
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_core.prompts import PromptTemplate

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.base import LLM
from langchain.chains import LLMChain

from llama_cpp import Llama

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model dir
model_dir = r"/home/dateng/model/huggingface/meta-llama/Llama-2-7b-chat-hf"
model_dir_gguf = '/home/dateng/model/huggingface/TheBloke/Llama-2-7B-Chat-GGUF'

"""
    CASE 1:
        Use Huggingface to call Llama2 (Llama-2-7b-chat-hf) model.
"""

# Model
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Prompt
prompt = "请给我讲个关于龙珠里的卡卡罗特的传说"

# Tokenization
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

"""
    CASE 2:
        Use LangChain and Huggingface to call Llama2 (Llama-2-7b-chat-hf) model.
"""

# Create pipeline base on Huggingface transformers and specified Model
pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=1000
)

# Create llm base on HuggingFacePipeline
llm_hf_pipeline = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 1})

# Prompt
template = """为以下的龙珠场景生成一个详细且吸引人的描述：
    龙珠的场景：{input}
"""
prompt_template = PromptTemplate(template=template, input_variables=["input"])

# LLM Chain
llm_chain = LLMChain(prompt=prompt_template, llm=llm_hf_pipeline)
result = llm_chain.invoke({"input": "卡卡罗特对战菲利萨"})

"""
    CASE 3:
        Use LangChain to call customized (llama-2-7b-chat.Q4_K_M.gguf) model.
"""

# Model name
model_name_gguf = 'llama-2-7b-chat.Q4_K_M.gguf'


# Custom LLM that inherited from the base LLM
class CustomizedLLM(LLM):
    name = model_name_gguf

    # Use the llama_cpp Llama library to call the quantized model to generate responses
    def _call(self, prompt: str, **kwargs: Any) -> str:
        llama_llm = Llama(model_path=os.path.join(model_dir_gguf, model_name_gguf), n_threads=4)
        response = llama_llm(f"Q: {prompt} A: ", max_tokens=256)
        output = response["choices"][0]["text"]
        return output

    @property
    def _llm_type(self) -> str:
        return "CustomizedLLM (LLM)"


# Customized llm
customized_llm = CustomizedLLM()
result = customized_llm.invoke("请给我讲个关于龙珠里的卡卡罗特的传说")
