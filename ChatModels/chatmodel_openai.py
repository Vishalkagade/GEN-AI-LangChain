from langchain_openai import ChatOpenAI

from dotenv import load_dotenv  # it loads the environment variables from the .env file

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=1.5, max_tokens=30)

result = model("write a 5 line poem on MS DHONI")  # invoke the model with the input prompt

print(result)
