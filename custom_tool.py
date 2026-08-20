from langchain.tools import tool

@tool
def get_greetings(name: str) -> str:
    """Generate a greeting message for the given name."""

    return f"Hello {name}, Welcome to the world of LangChain!"

result = get_greetings.invoke({"name": "Alice"})
print(result)

print(get_greetings.name)
print(get_greetings.description)
print(get_greetings.args)