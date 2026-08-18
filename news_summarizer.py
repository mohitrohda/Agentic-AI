from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_results=3)

llm = ChatMistralAI(model_name="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that summarize the following news in clear bullets points {news}.
"""
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("latest news about AI")

result = chain.invoke({"news": news_result})

print(result)