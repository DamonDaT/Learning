from dotenv import load_dotenv

from langchain.chains import LLMChain, SequentialChain

from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAI

# Initialize environment variables from .env file
load_dotenv(verbose=True)

llm = OpenAI(max_tokens=1024)

"""
    STEP 1:
        LLM Chain 1.
"""

# Prompt
template_1 = """你是一位动漫设计者。根据给定动漫的名称，你需要为这部动漫写一个100字左右的设计细节。
\n动漫名称: {cartoon_name}
\n动漫设计者: 这是关于上述动漫的设计细节:
"""
prompt_template_1 = PromptTemplate(template=template_1, input_variables=["cartoon_name"])

# LLM Chain
llm_chain_1 = LLMChain(llm=llm, prompt=prompt_template_1, output_key="design")

"""
    STEP 2:
        LLM Chain 2.
"""

# Prompt
template_2 = """你是一位动漫评论家。根据给定动漫的设计细节，你需要为这部动漫写一个100字左右的评论。
\n动漫设计细节: 
{design}
\n动漫评论家: 这是关于上述动漫的评论:
"""
prompt_template_2 = PromptTemplate(template=template_2, input_variables=["introduction"])

# LLM Chain
llm_chain_2 = LLMChain(llm=llm, prompt=prompt_template_2, output_key="review")

"""
    STEP 3:
        LLM Chain 3.
"""

# Prompt
template_3 = """你是一位动漫销售员。根据给定动漫的设计细节和评论，你需要为这部动漫写一个100字左右的销售文案。
\n动漫设计细节: 
{design}
\n动漫评论:
{review}
\n动漫销售文案:
"""
prompt_template_3 = PromptTemplate(template=template_3, input_variables=["design", "review"])

# LLM Chain
llm_chain_3 = LLMChain(llm=llm, prompt=prompt_template_3, output_key="sales_article")

"""
    STEP 4:
        SequentialChain.
"""

# Sequential chain
sequential_chain = SequentialChain(
    chains=[llm_chain_1, llm_chain_2, llm_chain_3],
    input_variables=["cartoon_name"],
    output_variables=["design", "review", "sales_article"],
    verbose=True
)

result = sequential_chain.invoke({"cartoon_name": "龙珠"})
