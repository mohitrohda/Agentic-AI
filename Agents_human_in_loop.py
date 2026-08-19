from dotenv import load_dotenv
load_dotenv()

import requests
import os

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient


# =========================
# 🌦️ Weather Tool
# =========================


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()
    
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    
    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool (Tavily)
# =========================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================
# 🧠 LLM Setup
# =========================

llm = ChatMistralAI(model_name="mistral-small-2506")

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tool = llm.bind_tools([get_weather, get_news])


# =========================
# Agent Loop
# =========================

messages = []

print("Welcome to The CITY INTELLIGENCE AGENT! Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Exiting the agent. Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tool.invoke(messages)

        messages.append(result)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]

                confirm = input(f"\nThe agent wants to call the tool '{tool_name}'. Do you want to proceed? (yes/no): ")

                if confirm.lower() == "no":
                    print("Tool call canceled by user.")
                    break

                tool_result = tools[tool_name].invoke(tool_call)

                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                    ))

            continue  # Continue the inner loop to process the next tool call if any

        else:
            print(f"\nAgent: {result.content}")
            break  # Exit the inner loop if no more tool calls are needed


            