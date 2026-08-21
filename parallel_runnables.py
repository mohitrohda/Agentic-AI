from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

#component
model = ChatMistralAI(model_name="mistral-small-2506")
parser = StrOutputParser()

#Two different prompts
prompt1 = ChatPromptTemplate.from_template(
    "Explain the {topic} in 1-2 lines"
)

prompt2 = ChatPromptTemplate.from_template(
    "Explain the {topic} in detail"
)

#input
topic = "Machine Learning"

chain = RunnableParallel({
    "short_explanation": RunnableLambda(lambda x:x ['short_explanation']) | prompt1 | model | parser,
    "detailed_explanation": RunnableLambda(lambda x:x ['detailed_explanation']) | prompt2 | model | parser
})

#result = chain.invoke({"topic": "Machine Learning"})

result = chain.invoke({
    "short_explanation": {"topic": "Machine Learning"},
    "detailed_explanation": {"topic": "deep learning"}
})

print(result["short_explanation"])
print(result["detailed_explanation"])