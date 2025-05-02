import yfinance as yf
from os import getenv
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

# Load environment variables
load_dotenv()
openai_api_key = getenv("OPENAI_API_KEY", "")

# Initialize the LLM
llm = OpenAI(openai_api_key=openai_api_key)

# Define the tool
def get_stock_price(stock_symbol: str) -> str:
    try:
        stock = yf.Ticker(stock_symbol)
        price = stock.history(period="1d")["Close"][0]
        return f"The current stock price of {stock_symbol.upper()} is ${price:.2f}."
    except Exception as e:
        return f"Error fetching stock price: {e}"

# Create a Tool object
tools = [
    Tool(
        name="get_stock_price",
        func=get_stock_price,
        description="Fetch the current stock price for a given symbol, e.g., AAPL"
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run a query
query = "What is the current stock price of AAPL?"
response = agent.run(query)
print(f"Agent Response: {response}")
