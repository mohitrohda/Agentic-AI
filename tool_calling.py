from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

from rich import print

# creating tool

@tool
def get_text_length(text: str) -> int:
    """Get the length of the text."""
    return len(text)

llm = ChatMistralAI(model_name="mistral-small-2506")

# tool binding

llm_with_tool = llm.bind_tools([get_text_length])

result = llm_with_tool.invoke("Get the length of the text - 'Hello, how are you?'")

#print(result.tool_calls[0])

if result.tool_calls:
    tool_call = result.tool_calls[0]

tool_name = tool_call["name"]
tool_args = tool_call["args"]

tool_result = get_text_length.invoke(tool_args)

final_result = llm_with_tool.invoke(f"The length of the text is {tool_result}")

print(final_result)















'''

result = llm.invoke("hello")
result2 = llm_with_tool.invoke("hello")

print(result)
print(result2)

# tool calling

result = llm.invoke("Get the length of the text - 'Hello, how are you?'")
result2 = llm_with_tool.invoke("Get the length of the text - 'Hello, how are you?'")

print(result)
print(result2)

'''

# tool execution
