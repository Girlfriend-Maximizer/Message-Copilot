from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq LLM
try:
    llm = ChatGroq(model="llama3-70b-8192")
except Exception as e:
    raise ValueError(f"Failed to initialize ChatGroq: {e}")

# Define the system prompt (initial context or instructions for the model)
with open("./prompts/gf_base", "r") as f:
    system_prompt = f.read()

# Chatbot loop
print("Welcome to your personal girlfriend messaging experience!")
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
        chain = chat_prompt | llm | StrOutputParser()
        
        # Use the chain to invoke the model and get the response
        response = chain.invoke({})
        
        # Print the response from the chain
        print("GF:", response)
        
        # Add the chatbot's message to the conversation history as a tuple
        conversation_history.append(("ai", response))
    
    except Exception as e:
        print("Error:", e)
