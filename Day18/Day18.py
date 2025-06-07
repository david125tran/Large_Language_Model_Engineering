# ------------------------------------ RAG (Retrieval Augmented Generation) ------------------------------------
# --- 1. Introduction to RAG ---
# RAG is an AI framework that enhances Large Language Models (LLMs)
# by giving them access to external, authoritative knowledge bases.
# This allows LLMs to generate more accurate, relevant, and up-to-date responses.

# --- 2. The Problem RAG Solves ---
# Traditional LLMs have limitations:
# a. Static Training Data: Knowledge is limited to their training data (can be outdated).
# b. Hallucinations: May generate incorrect or made-up information for new topics.
# c. Lack of Specificity: Struggle with domain-specific or internal company data.
# d. Cost of Retraining: Expensive and time-consuming to update LLMs frequently.

# --- 3. How RAG Works (The Process) ---
# RAG combines two main phases: Retrieval and Generation.

# --- Phase A: Ingestion/Indexing (Pre-processing) ---
# Goal: Prepare and store the external knowledge base.
# Steps:
# 1. External Data Preparation: Define and gather data (documents, databases, APIs, etc.).
# 2. Chunking: Break down large documents into smaller, manageable "chunks" or segments.
#    # Example: A long PDF broken into paragraphs or fixed-size text blocks.
# 3. Embedding: Convert each chunk into a numerical representation (vector embedding).
#    # These embeddings capture the semantic meaning of the text.
# 4. Vector Database Storage: Store these vector embeddings in a specialized database
#    # (e.g., Pinecone, Weaviate, ChromaDB) optimized for fast similarity searches.

# --- Phase B: Retrieval Phase ---
# Goal: Find the most relevant information from the knowledge base for a given query.
# Steps:
# 1. User Query: A user asks a question or provides a prompt.
# 2. Query Embedding: The user's query is also converted into a vector embedding.
# 3. Semantic Search: The system performs a similarity search in the vector database
#    # comparing the query's embedding to the stored chunk embeddings.
#    # This finds conceptually similar information, not just keyword matches.
# 4. Re-ranking (Optional): Further refines retrieved documents to ensure top relevance.

# --- Phase C: Augmented Generation Phase ---
# Goal: Use the retrieved information to enhance the LLM's response.
# Steps:
# 1. Prompt Augmentation: The original user query is combined with the top N
#    # most relevant retrieved chunks. This is often called "prompt stuffing."
#    # Example: "User query: [original_query]\n\nContext: [retrieved_chunk_1]\n[retrieved_chunk_2]..."
# 2. LLM Generation: The augmented prompt is fed into the LLM.
#    # The LLM uses its pre-trained knowledge *and* the provided context.
# 3. Response Delivery & Citation: The LLM generates a more accurate response.
#    # Ideally, RAG systems also provide sources/citations from the retrieved documents
#    # for verification and trust.

# --- 4. Key Benefits of RAG ---
# 1. Factuality & Reduced Hallucinations: Grounds LLM in verifiable external data.
# 2. Up-to-Date Information: Accesses the latest data without LLM retraining.


# ------------------------------------ RAG vs. Tooling ----------------------------------
# Key Differences: RAG vs. Tooling/Function Calling

# --- Introduction ---
# Both RAG (Retrieval Augmented Generation) and Tooling (or Function Calling)
# empower LLMs with external data, but they differ fundamentally in *how*
# that data is integrated and utilized.

# --- Table of Differences ---
# Feature             | RAG (Retrieval Augmented Generation)        | Tooling / Function Calling (API Feeding)
# --------------------|---------------------------------------------|----------------------------------------------------------
# 1. Core Purpose     | Ground LLM generation in factual context.   | Enable LLM to interact with external systems; perform actions.
# 2. Data Flow        | Data is **retrieved first**, then **fed**   | LLM **decides to call a tool**, tool **fetches data/performs action**,
#                     | **into LLM's prompt as context**.           | result **fed back to LLM for final response**.
# 3. LLM's Role       | Context consumer; generates response based  | Agent; decides *which external action to take* to fulfill user intent.
#                     | on provided text.                           |
# 4. Primary Data Type| Primarily unstructured text (documents,     | Structured data from APIs, databases; can also be actions/commands.
#                     | articles, PDFs) for semantic search.        |
# 5. Interaction Model| "Here's some information, now answer."      | "I need to do X, so I'll use Tool Y to get Z, then answer."
# 6. Implementation   | Typically simpler for information retrieval.| Can be more complex due to tool orchestration, state management,
#   Complexity        |                                             | and error handling.
# 7. Real-time Access | Yes, for the knowledge base it uses (which  | Yes, often for live, dynamic data or actions (e.g., current weather, sending emails).
#                     | can be updated frequently).                 |

# --- Analogy ---
# RAG: LLM is like a student with an open textbook (retrieved context) for an exam.
# Tooling: LLM is like a smart assistant who knows how to use various tools (weather app, calculator) to fulfill requests.

# --- Can they be combined? ---
# Yes, absolutely! Advanced LLM applications often leverage both.
# Example:
# User: "What's our company's policy on remote work, and what's the current weather in London?"
# - RAG for "remote work policy" (retrieves from internal documents).
# - Tooling for "current weather in London" (calls weather API).
# - LLM then combines both pieces of information into a single coherent response.


# ------------------------------------ Packages ----------------------------------
# pip install anthropic
# pip install dotenv
# pip install gradio


# ------------------------------------ Imports ----------------------------------
import os
import glob
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI


# ------------------------------------ Configure API Keys / Tokens ----------------------------------
# Specify the path to the .env file
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

# Access the API keys stored in the environment variable
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')            # https://openai.com/api/
# anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')      # https://console.anthropic.com/ 
# google_api_key = os.getenv('GOOGLE_API_KEY')            # https://ai.google.dev/gemini-api
# huggingface_token = os.getenv('HUGGINGFACE_TOKEN')      # https://huggingface.co/settings/tokens


print("Checking API Keys...\n")
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:10]}")
else:
    print("OpenAI API Key not set")
    
# if anthropic_api_key:
#     print(f"Anthropic API Key exists and begins {anthropic_api_key[:10]}")
# else:
#     print("Anthropic API Key not set")

# if google_api_key:
#     print(f"Google API Key exists and begins {google_api_key[:10]}")
# else:
#     print("Google API Key not set")

# if huggingface_token:
#     print(f"Hugging Face Token exists and begins {huggingface_token[:10]}")
# else:
#     print("Hugging Face Token not set")
print("\n------------------------------------\n")


# ------------------------------------ Connect to API Platforms ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# Lower cost model:
OPENAI_MODEL = "gpt-4o-mini"

# load_dotenv(override=True) # This is already done above, no need to repeat here
openai = OpenAI()

# ------------------------------------ Create Employee Dictionary ----------------------------------
# Initialize the context dictionary to store employee and product information
context = {}
# Example Structure:
# context = {
#     "Avery Lancaster": "Avery is a senior software engineer at Insurellm...",
#     "CarLLM": "CarLLM is a product of Insurellm that provides...",
#     "Samuel Trenton": "Samuel is a data scientist at Insurellm...",
#     "Product X": "Product X is an insurance product that...",
# }


# ------------------------------------ Load Employee Files ----------------------------------
# Look for employee files located in: "C:\Users\Laptop\Desktop\Coding\LLM\Day18\knowledge-base\employees\<First Name Last Name>.md"
employees_dir = os.path.join(script_dir, "knowledge-base", "employees")
# Use glob to find all markdown files in the employees directory.  Ex. "Samuel Trenton.md"
employees_files = glob.glob(os.path.join(employees_dir, "*.md"))

# For each employee file, extract the employee name and content
for employee_file_path in employees_files:
    # Extract the filename from the full path
    filename = os.path.basename(employee_file_path)
    # Ex. Extracts "Avery Lancaster" from "Avery Lancaster.md"
    name = os.path.splitext(filename)[0] 
    # Initialize doc variable to store the content of the employee file
    doc = ""
    # Open the employee file and read its content
    with open(employee_file_path, "r", encoding="utf-8") as f:
        doc = f.read()
    # Store in context dictionary
    context[name] = doc 

# ------------------------------------ Load Product Files ----------------------------------
# Look for product files located in: "C:\Users\Laptop\Desktop\Coding\LLM\Day18\knowledge-base\products\<product name>.md"
products_dir = os.path.join(script_dir, "knowledge-base", "products")
products_files = glob.glob(os.path.join(products_dir, "*.md")) # Assuming products are also .md files, adjust if .txt

# For each product file, extract the product name and content
for product_file_path in products_files:
    # Extract the filename from the full path
    filename = os.path.basename(product_file_path) 
    # Extracts "CarLLM" from "CarLLM.md"
    name = os.path.splitext(filename)[0] 
    # Initialize doc variable to store the content of the product file
    doc = ""
    # Open the product file and read its content
    with open(product_file_path, "r", encoding="utf-8") as f:
        doc = f.read()
    # Store in context dictionary
    context[name] = doc

# ------------------------------------ Debugging Context Keys ----------------------------------
# Print all context keys for debugging purposes
# print("All context keys:", context.keys()) 


# ------------------------------------ System Message  ----------------------------------
system_message = """You are an expert in answering accurate questions about Insurellm, the Insurance 
Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything 
up if you haven't been provided with relevant context."""


# ------------------------------------ Helper Functions for System Prompt  ----------------------------------
def get_relevant_context(message):
    """
    This function retrieves relevant context from the context dictionary based on the user's message.
    It checks if the context title (e.g., "Avery Lancaster", "CarLLM") is present in the message.
    Args:
        message (str): The user's message or question.
    Returns:
        list: A list of relevant context strings that match the message.
    """
    # Initialize an empty list to store relevant context
    relevant_context = []
    # Lowercase the message once for efficiency
    lower_message = message.lower()
    # Iterate through each context title and its details in the context dictionary
    for context_title, context_details in context.items():
        # Check if the context_title (e.g., "avery lancaster" in "who is avery lancaster?") is found in the message.  
        if context_title.lower() in lower_message: 
            # If found, append the context details (contents of *.md file) to the relevant_context list
            relevant_context.append(context_details)

    # If no relevant context is found, return an empty list

    # If found, it returns a list of extracted context details.  I.e. a list containing each line of the
    # markdown file for the relevant context. 
    # Ex. Contents of "Avery Lancaster.md" from "C:\Users\Laptop\Desktop\Coding\LLM\Day18\knowledge-base\employees\Avery Lancaster.md"
    return relevant_context  


def add_context(message):
    """
    This function adds relevant context to the user's message by appending any relevant context 
    retrieved from the context dictionary.
    Args:
        message (str): The user's message or question.
    Returns:
        str: The original message with appended relevant context, if any.
    """
    # Call get_relevant_context() to retrieve context based on the user's message
    # If relevant context is found, append it to the message
    # If no relevant context is found, the message remains unchanged
    relevant_context = get_relevant_context(message)
    # If relevant context is found, append it to the message
    # If no relevant context is found, the message remains unchanged
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"

    # If found, it appends the relevant context to the message.
    # If not found, it returns the original message unchanged.

    # Example Return: 
    # Who is Avery Lancaster?

    # The following additional context might be relevant in answering this question:

    # # Avery Lancaster

    # ## Summary
    # - **Date of Birth**: March 15, 1985
    # - **Job Title**: Co-Founder & Chief Executive Officer (CEO)
    # - **Location**: San Francisco, California
    # ... 
    return message


# ------------------------------------ Chat Function for Gradio Interface ----------------------------------
def chat(message, history):
    """
    This function handles the chat interaction with the OpenAI API.
    It prepares the messages, adds relevant context, and streams the response.
    Args:
        message (str): The user's message or question.
        history (list): The chat history, which is a list of tuples containing previous messages.
                        Each tuple contains (human_message, ai_message).
    Returns:
        generator: A generator that yields the response from the OpenAI API in chunks.  
    """
    # Start with the system message
    messages = [{"role": "system", "content": system_message}]

    # Convert Gradio's history format to OpenAI's messages format
    for human_message, ai_message in history:
        messages.append({"role": "user", "content": human_message})
        messages.append({"role": "assistant", "content": ai_message})

    # Add the user's current message (augmented with context)
    message = add_context(message)
    messages.append({"role": "user", "content": message})

    # Create a streaming response from the OpenAI API
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages, stream=True)

    # Initialize an empty string to accumulate the response
    # The response will be built up as chunks are received from the OpenAI API
    # This allows for real-time updates to the response as each chunk is received
    # The response will be yielded in chunks, allowing the Gradio interface to display it incrementally
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response


# ------------------------------------ Testing the get_relevant_context() Function ----------------------------------
print("\n--- Testing get_relevant_context() ---\n")
# This will successfully retrieve context for 'Avery Lancaster'
print(" ----- Relevant context for 'Who is Avery Lancaster?': ----- \n", get_relevant_context("Who is Avery Lancaster?"), "\n")

# This will not work because "avery lancaster" is not in "lancaster"
print(" ----- Relevant context for 'Who is lancaster?': ----- \n", get_relevant_context("Who is lancaster?"), "\n") 

# This will not work for "Avery" (because "avery lancaster" is not in "lancaster") but will successfully retrieve context for 'CarLLM'
print(" ----- Relevant context for 'Who is Avery and what is carllm?': ----- \n", get_relevant_context("Who is Avery and what is carllm?"), "\n")

# This will successfully retrieve context for 'Avery Lancaster' and 'CarLLM'
print(" ----- Relevant context for 'Who is Avery Lancaster and what is carllm?': ----- \n", get_relevant_context("Who is Avery Lancaster and what is carllm?"), "\n")


# ------------------------------------ Testing the add_context() Function ----------------------------------
print("\n--- Testing add_context ---\n")
# This wil successfully retrieve context for 'Avery Lancaster' and append it to the message
print(" ----- add_context('Who is Avery Lancaster?') ----- \n", add_context("Who is Avery Lancaster?"), "\n")


# ------------------------------------ Gradio Interface ----------------------------------
if __name__ == "__main__":
   gr.ChatInterface(chat).launch()