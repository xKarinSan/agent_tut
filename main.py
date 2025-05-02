from os import getenv
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

openai_api_key = getenv("OPENAI_API_KEY","")
# Initialize OpenAI LLM
llm = OpenAI(openai_api_key=openai_api_key)

# Define a simple prompt for the agent
template = """
You are an AI assistant with expertise in data analysis and automation. Answer the following question:
Question: {question}
"""

# Set up the prompt and LLM chain
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)

# Example query
query = "What is the impact of AI in healthcare?"
response = chain.run(question=query)
print(f"Agent Response: {response}")