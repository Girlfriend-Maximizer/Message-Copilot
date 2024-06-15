import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Get the Groq API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Check if API key is loaded
if api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file or environment settings.")

# Initialize the Groq LLM with proper error handling
try:
    llm = ChatGroq(api_key=api_key, model="llama3-70b-8192")
except Exception as e:
    raise ValueError(f"Failed to initialize ChatGroq: {e}")

# Initialize the output parser
output_parser = StrOutputParser()

# Define the system prompt (initial context or instructions for the model)
system_prompt = "You are a helpful assistant trained to act like the girlfriend conversing with their significant other."

# Chatbot loop
print("Welcome to the LLaMA3-70b Chatbot powered by Groq!")
print("Type 'quit', 'exit', or 'bye' to exit.")

# Maintain a list of conversation history
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Exiting the chat. Goodbye!")
        break
    
    # Add the user's message to the conversation history as a tuple
    conversation_history.append(("human", user_input))
    
    try:
        # Create a ChatPromptTemplate from the system prompt and conversation history
        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt)] + conversation_history
        )
        
        # Create a chain from the prompt template and the model
        chain = chat_prompt | llm | output_parser
        
        # Use the chain to invoke the model and get the response
        response = chain.invoke({})
        
        # Print the response from the chain
        print("Chatbot:", response)
        
        # Add the chatbot's message to the conversation history as a tuple
        conversation_history.append(("ai", response))
    
    except Exception as e:
        print("Error:", e)
