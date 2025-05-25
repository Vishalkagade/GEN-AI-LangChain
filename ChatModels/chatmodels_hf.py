from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Create the HuggingFace endpoint with token
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",  # <- Correct task for chat models
    huggingfacehub_api_token=hf_token
)

# Wrap with LangChain chat interface
model = ChatHuggingFace(llm=llm)

# Invoke the model
result = model.invoke("What is the capital of India?")
print(result)

