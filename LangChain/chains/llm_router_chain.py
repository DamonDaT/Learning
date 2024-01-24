from dotenv import load_dotenv

from langchain_openai import OpenAI

from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as ROUTER_TEMPLATE

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model
llm = OpenAI()

# Template
cartoon_scene_template = """你是一个看过过众多动漫的动漫爱好者，擅长解答描述动漫情节的问题。
下面是需要你来回答的问题:
{input}
"""

cartoon_sale_template = """你是一位资深的动漫销售员，擅长解答关于动漫销售的问题。
下面是需要你来回答的问题:
{input}
"""

prompt_infos = [
    {
        "name": "cartoon_scene",
        "description": "适合回答关于动漫情节的问题",
        "prompt_template": cartoon_scene_template,
    },
    {
        "name": "cartoon_sale",
        "description": "适合回答关于动漫销售的问题",
        "prompt_template": cartoon_sale_template,
    }
]

# Prompt (Create destinations for ROUTER_TEMPLATE)
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
router_prompt = ROUTER_TEMPLATE.format(destinations="\n".join(destinations))
router_prompt_template = PromptTemplate(
    template=router_prompt,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)

# LLM router chain (The top-level chain that is used to decide which destination or default chain to call)
llm_router_chain = LLMRouterChain.from_llm(llm, router_prompt_template, verbose=True)

# LLM Chains (dict. Use as destination chains)
llm_chains = {}
for prompt_info in prompt_infos:
    prompt_template = PromptTemplate(template=prompt_info["prompt_template"], input_variables=["input"])
    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    llm_chains[prompt_info["name"]] = llm_chain

# Conversation chain (Use as default chain)
conversation_chain = ConversationChain(llm=llm, output_key="text", verbose=True)

# Multi prompt chain
multi_prompt_chain = MultiPromptChain(
    router_chain=llm_router_chain,
    destination_chains=llm_chains,
    default_chain=conversation_chain,
    verbose=True
)

multi_prompt_chain = MultiPromptChain.from_prompts(
    llm=llm,
    prompt_infos=prompt_infos,
    default_chain=conversation_chain,
    verbose=True
)

result_1 = multi_prompt_chain.invoke({"input": "龙珠怎么卖会比较好呢？"})
result_2 = multi_prompt_chain.invoke({"input": "卡卡罗特对战贝吉塔的场景是怎样的？"})
result_3 = multi_prompt_chain.invoke({"input": "如何理解马斯克的时间拳击法？"})
