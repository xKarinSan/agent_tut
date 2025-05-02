from os import getenv
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from pandas import DataFrame

# Load environment variables
load_dotenv()
openai_api_key = getenv("OPENAI_API_KEY", "")

# Initialize the LLM
llm = OpenAI(openai_api_key=openai_api_key)

# Create a small dataset
data = {
    'Product': ['Laptop', 'Smartphone', 'Tablet', 'Desktop'],
    'Sales': [150, 200, 100, 90]
}
df = DataFrame(data)

# Define a string-accepting analysis tool
def analyze_sales(_: str) -> str:
    max_sales = df.loc[df['Sales'].idxmax()]
    return f"Top-selling product is {max_sales['Product']} with {max_sales['Sales']} units."

# Wrap the function as a Tool
analysis_tool = Tool(
    name="analyze_sales",
    func=analyze_sales,
    description="Analyze sales data and provide insights about top-selling products."
)

# Initialize the agent with tool
agent = initialize_agent(
    tools=[analysis_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run a multi-role query
query = "Analyze the sales data and suggest business strategies."
response = agent.run(query)
print(f"Agent Response: {response}")
