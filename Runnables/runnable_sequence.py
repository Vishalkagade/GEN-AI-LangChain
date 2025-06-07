from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from langchain.schema.runnable import RunnableSequence


load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(template="create a funny joke about {topic}",
                        input_variables=["topic"])

parser = StrOutputParser()

prompt2 = PromptTemplate(template="give the counter if the following Joke {text}",
                         input_variables="text")

chain = RunnableSequence(prompt, model, parser,prompt2,model,parser)

result = chain.invoke({"topic": "Virat Kohli"})

print(result)