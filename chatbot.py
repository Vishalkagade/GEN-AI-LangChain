from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
chat_history = [
    SystemMessage(content = "You are a helpful AI Assistant"),
    
                ]

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))

    print("AI", result.content)
print(chat_history)
