from os import getenv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from pydantic import BaseModel

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


openai_api_key = getenv("OPENAI_API_KEY", "")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# response =llm.invoke("What is the meaning of life?")
# print(response)

# parse the pydantic object -> uses basemodel
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that generates a research paper.
            Answer the user query and use the necessary tools.
            Wrap the output in this format and provide no other text \n {format_instructions}
            """
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())
# uses parser and the pydantic model and turn to string

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools = []
)

# verbose -> display thought process
agent_executor = AgentExecutor(agent=agent, tools = [], verbose=True)
raw_response = agent_executor.invoke({"query":"What is the capital of France?"})
print(raw_response)
# parse:
structured_response = parser.parse(raw_response.get("output"))
print(f"Structured response: \n {structured_response.topic}")