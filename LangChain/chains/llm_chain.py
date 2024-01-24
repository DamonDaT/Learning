from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_openai.llms import OpenAI

from langchain.chains import LLMChain

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Prompt
template = "简单介绍下: {cartoon_name}"
prompt_template = PromptTemplate.from_template(template)
prompt_str = prompt_template.format(cartoon_name="龙珠")

# Multiple inputs
input_list = [
    {"cartoon_name": "龙珠"},
    {"cartoon_name": "海贼王"}
]

# Model
llm = OpenAI()

# LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

"""
    CASE 0:
        Traditional modelio process.
"""

result = llm.invoke(prompt_str)

"""
    CASE 1:
        LLMChain (invoke).
    Return:
        dict. {"variable": "xxx", "text": "yyy"}
"""

result = llm_chain.invoke({"cartoon_name": "龙珠"})

"""
    CASE 2:
        LLMChain (predict).
    Return:
        str.
"""

result = llm_chain.predict(cartoon_name="龙珠")

"""
    CASE 3:
        LLMChain (apply).
    Return:
        list. [{"text": "xxx"}, {"text": "yyy"}]
"""

result = llm_chain.apply(input_list=input_list)

"""
    CASE 4:
        LLMChain (generate).
    Return:
        langchain_core.outputs.llm_result.LLMResult
"""

result = llm_chain.generate(input_list)
