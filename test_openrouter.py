from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables (make sure you have gptOSS in .env)
load_dotenv()
api_key = os.getenv("gptOSS")

if not api_key:
    raise ValueError("Missing API key: please set gptOSS in your .env file")

# Initialize GPT-OSS model from OpenRouter
llm = ChatOpenAI(
    model="gpt-oss-120b",   # free GPT-OSS model
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)

# Create a simple conversation
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello, can you confirm my GPT-OSS key works?")
]

# Invoke the model
response = llm.invoke(messages)

print("✅ Response from model:")
print(response.content)
