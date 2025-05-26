from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template= "Generate detailed report on the {topic}",
    input_variables= ["topic"]
)

prompt2 = PromptTemplate(
    template= "Generate 5 most important point from following quary \n {quary}",
    input_variables= ["quary"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic":"Job Market situation in germany"})
print(result)
