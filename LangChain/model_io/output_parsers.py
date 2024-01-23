from dotenv import load_dotenv
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    CASE 1:
        PydanticOutputParser(Json).
"""


class DragonBallScene(BaseModel):
    role1: str = Field(description="角色1")
    role2: str = Field(description="角色2")
    scene_type: str = Field(description="场景类别")
    description: str = Field(description="场景描述")


# Pydantic output parser
pydantic_output_parser = PydanticOutputParser(pydantic_object=DragonBallScene)

# Format instructions
format_instructions = pydantic_output_parser.get_format_instructions()
# Prompt
template = """您是一位资深的龙珠动漫研究员。
对于{role1}对战{role2}的场景，您能提一个激动人心的场景细节描述吗？
\n{format_instructions}"""
prompt_template = PromptTemplate(
    template=template,
    input_variables=["role1", "role2"],
    partial_variables={"format_instructions": format_instructions}
)
prompt_str = prompt_template.format(role1="卡卡罗特", role2="贝吉塔")

# Model
model = OpenAI(model_name="gpt-3.5-turbo-instruct", max_tokens=1024)
result = model.invoke(prompt_str)

"""
    CASE 2:
        OutputFixingParser.
"""


class DragonBallRoles(BaseModel):
    protagonist: str = Field(description="主角")
    supporting_roles: List[str] = Field(description="配角")


# Incorrect formatted (Double quotes must be inside single quotes)
incorrect_formatted = "{'protagonist': '卡卡罗特', 'supporting_roles': ['弗利萨','沙鲁','布欧']}"

# Output parser
pydantic_output_parser = PydanticOutputParser(pydantic_object=DragonBallRoles)
output_fixing_parser = OutputFixingParser.from_llm(parser=pydantic_output_parser, llm=ChatOpenAI())

result = output_fixing_parser.parse(incorrect_formatted)

"""
    CASE 3:
        RetryWithErrorOutputParser.
"""


class CartoonAnimation(BaseModel):
    name: str = Field(description="动漫名称")
    animation: str = Field(description="动效设计")


# Incorrect formatted (One value (forest) is missing)
incorrect_formatted = '{"name": "龙珠"}'

# Output parser
pydantic_output_parser = PydanticOutputParser(pydantic_object=CartoonAnimation)
retry_parser = RetryWithErrorOutputParser.from_llm(parser=pydantic_output_parser, llm=ChatOpenAI())

# Format instructions
format_instructions = pydantic_output_parser.get_format_instructions()
# Prompt
template = """您是一位资深的动漫动效设计师，根据给定的动漫名称设计相应的动效。
\n{format_instructions}
\nCartoon name: {name}
Response:"""
prompt_template = PromptTemplate(
    template=template,
    input_variables=["name"],
    partial_variables={"format_instructions": format_instructions}
)
prompt_value = prompt_template.format_prompt(name="龙珠")

# Parse with prompt (Notice: Incompatibility between pydantic v1 and v2)
result = retry_parser.parse_with_prompt(incorrect_formatted, prompt_value)
