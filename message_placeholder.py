from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# Chat Template

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful AI Assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

chat_history = []

with open("chat_history.txt", "r") as file:
    chat_history.extend(file.readlines())

print("Chat History:", chat_history)
# Add a new message to the chat history

prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        'query':'Where is my refund'
    }
)

print("Prompt:", prompt)