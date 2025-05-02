from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool 
from datetime import datetime

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="searchWeb",
    func=search.run,
    # desc -> when to use the tool
    description="Search the web for information"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_characters=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)