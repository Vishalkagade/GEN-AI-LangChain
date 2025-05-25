from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Wrap with LangChain chat interface
model = ChatOpenAI()

template1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(template= "write a five line summary of {text}",
    input_variables=["text"]

)

# Create an output parser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

# prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
# result1 = model.invoke(prompt1)
# prompt2 = template2.invoke({"text": result1.content})
# result2 = model.invoke(prompt2)

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)