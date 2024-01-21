from dotenv import load_dotenv

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotPromptTemplate
)

from langchain_openai import (
    OpenAI,
    ChatOpenAI
)

from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    CASE 1:
        PromptTemplate.
"""

# Prompt
template = "您是一位专业的鲜花店文案撰写员。\n对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？"
prompt_template = PromptTemplate.from_template(template)
prompt_str = prompt_template.format(flower_name="玫瑰", price="50")

# Model
model = OpenAI()
result = model.invoke(prompt_str)

"""
    CASE 2:
        ChatPromptTemplate / SystemMessagePromptTemplate / HumanMessagePromptTemplate.
"""

# Prompt
system_template = "你是一位专业顾问，负责为专注于{product}的公司起名。"
system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_template)
human_template = "公司主打产品是{product_detail}。"
human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt_template, human_message_prompt_template])
chat_prompt_value = chat_prompt_template.format_prompt(product="鲜花装饰",
                                                       product_detail="创新的鲜花设计。").to_messages()

# Model
chat_model = ChatOpenAI()
result = chat_model.invoke(chat_prompt_value).content

"""
    CASE 3:
        FewShotPromptTemplate.
"""

examples = [
    {
        "flower_type": "玫瑰",
        "occasion": "爱情",
        "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
    },
    {
        "flower_type": "康乃馨",
        "occasion": "母亲节",
        "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
    },
    {
        "flower_type": "百合",
        "occasion": "庆祝",
        "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
    },
    {
        "flower_type": "向日葵",
        "occasion": "鼓励",
        "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
    }
]

# Prompt
template = "鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}"
prompt_template = PromptTemplate(template=template, input_variables=["flower_type", "occasion", "ad_copy"])

# Example Selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=1
)

few_shot_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt_template,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"]
)

prompt_str = few_shot_prompt_template.format(flower_type="野玫瑰", occasion="爱情")

# Model
model = OpenAI()
result = model.invoke(prompt_str)
