import os
import requests

from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Proxy
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# OpenAI Key
os.environ['OPENAI_API_KEY'] = 'xxx'

# Image caption model from Hugging Face
model_dir = '/home/dateng/file/huggingface/Salesforce_blip-image-captioning-large/'

# Initialize processor
processor = BlipProcessor.from_pretrained(model_dir)

# Initialize model
model = BlipForConditionalGeneration.from_pretrained(model_dir)


# Custom tools
class ImageCapTool(BaseTool):
    name = "Image Caption"
    description = "Create captions for images."

    def _run(self, url: str):
        # Download the image and convert it to a PIL object
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # Change image into pixel values
        inputs = processor(image, return_tensors="pt")
        # Generate subtitles
        out = model.generate(**inputs, max_new_tokens=20)
        # Decode subtitles
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


# LLM（OpenAI）
llm = ChatOpenAI(model_name='gpt-4', temperature=0.8)

# Tools
tools = [ImageCapTool()]

# Agent
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
)

# A picture of flowers with the word love
img_url = 'https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg'

agent_input = f"{img_url}\n请给出浪漫的中文文案"

agent.run(input=agent_input)

"""
> Entering new AgentExecutor chain...
这不是一个直接的问题，而是一个对图像的请求，要求提供一个浪漫的中文文案。我需要先查看图像，理解其内容，然后才能创建合适的标题。
Action: Image Caption
Action Input: https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg
Observation: there is a picture of a picture of flowers with the word love
Thought:图像显示了一幅写着“爱”的花朵图。这个图像具有很强的浪漫气息，因为花朵和“爱”这个词都与浪漫紧密相关。现在我可以根据这个图像创作一个浪漫的中文文案。
Final Answer: 那些花儿，绽放的热烈，如同我的爱，永不凋零。

> Finished chain.
"""
