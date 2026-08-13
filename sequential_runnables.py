from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in simple terms."
)

model = ChatMistralAI(model_name="mistral-small-2506")

parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke("Machine Learning")  

print(result) 



'''formatted_prompt = prompt.format_prompt(topic="the theory of relativity")

response = model.invoke(formatted_prompt)

final_output = parser.parse(response)

print(final_output.content) '''