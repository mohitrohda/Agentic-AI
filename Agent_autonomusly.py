from dotenv import load_dotenv
load_dotenv()

import requests
import os

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


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

@wrap_tool_call
def human_approval(request, handler):
    """Middleware to ask for human approval before executing a tool."""

    tool_name = request.tool_call["name"]
    confirm = input(f"Do you want to call the tool '{tool_name}'? (yes/no): ").strip().lower()

    if confirm != "yes":
        return ToolMessage(
            content = "Tool call approved by human.",
            tool_call_id = request.tool_call["id"]
        )

    return handler(request)  # Proceed without calling the tool if not approved

agent = create_agent(
    llm,
    tools = [get_weather, get_news],
    system_prompt = "You are a helpful assistant that can provide weather updates and news about cities. Use the tools provided to fetch the information when needed.",
    middleware = [human_approval]
)

print("City Interaction Agent is ready. Type 'exit' to quit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Exiting the agent. Goodbye!")
        break

    result = agent.invoke({
        "messages" : [{"role": "user", "content": user_input}]
    })


    print(f"Agent: {result['messages'][-1].content}")