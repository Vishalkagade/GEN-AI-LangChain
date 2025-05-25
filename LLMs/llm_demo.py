from langchain_openai import OpenAI
from dotenv import load_dotenv # it loads the environment variables from the .env file

load_dotenv()


llm = OpenAI(model = "gpt-3.5-turbo-instruct") # you can change the model and temperature as per your requirement

result = llm.invoke("What is capital of Austria") # this will start the LLM and you can interact with it

print(result)


