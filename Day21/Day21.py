# This script demonstrates a **Retrieval-Augmented Generation (RAG)** system 
# for building a conversational AI chatbot. It combines the power of Large Language Models (LLMs)
# with a knowledge base to provide informed and context-aware responses.
#
# At a high level, this script performs the following key steps:
#
# 1.  **Document Ingestion & Preparation**:
#     It loads unstructured text documents (Markdown files) from a specified "knowledge-base" folder.
#     These documents are then systematically split into smaller, overlapping chunks to optimize
#     for retrieval accuracy and LLM context window limits.
#
# 2.  **Vector Embedding & Storage**:
#     Each text chunk is transformed into a numerical representation called a "vector embedding"
#     using OpenAI's embedding model. These embeddings capture the semantic meaning of the text.
#     The embeddings are stored in a **FAISS vector store**, enabling efficient
#     similarity searches to find relevant information.
#
# 3.  **Vector Space Visualization**:
#     To understand the relationships between the embedded documents, the script uses
#     t-SNE (t-Distributed Stochastic Neighbor Embedding) to reduce the high-dimensional
#     vectors into 2D and 3D spaces. These visualizations help show how similar documents
#     cluster together based on their meaning.
#
# 4.  **LLM Integration with RAG**:
#     It sets up a **Conversational Retrieval Chain** using LangChain. This chain
#     connects an OpenAI Chat LLM (like `gpt-4o-mini`), the FAISS vector store as a retriever,
#     and conversational memory. This integration allows the LLM to:
#     * **Retrieve**: Find relevant document chunks from the vector store based on user queries.
#     * **Augment**: Include the retrieved information in the LLM's prompt, providing context.
#     * **Generate**: Create accurate and coherent responses, maintaining conversational history.
#
# 5.  **Gradio User Interface**:
#     Finally, it launches a simple, interactive chat interface using Gradio, allowing users
#     to interact with the RAG chatbot and experience its ability to answer questions
#     based on the provided knowledge base while remembering past interactions.
#
# This setup provides a robust foundation for building chatbots that can answer questions
# beyond their initial training data by leveraging external, domain-specific knowledge.


# ------------------------------------ Packages ----------------------------------
# pip install anthropic
# pip install chromadb
# pip install dotenv
# pip install faiss-cpu
# pip install langchain
# pip install -U langchain-chroma
# pip install -U langchain-community
# pip install langchain-openai
# pip install openai
# pip install python-dotenv
# pip install scikit-learn


# ------------------------------------ Imports ----------------------------------
import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import FAISS  # Facebook AI Similarity Search
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


# ------------------------------------ Constants / Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
pio.renderers.default = 'browser'  # Or try 'chrome', 'firefox' explicitly
MODEL = "gpt-4o-mini"


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


# ------------------------------------ Read in Documents using LangChain's Loaders ----------------------------------
def load_documents_from_folder(folder_path):
    # Define the text loader kwargs
    text_loader_kwargs = {'encoding': 'utf-8'}

    # Initialize a list to hold all documents
    documents = []
    # Iterate through each folder found in the knowledge base
    for folder in folder_path:
        # Ensure 'folder' is indeed a directory before processing
        if os.path.isdir(folder):
            # Extract the document type from the folder name
            # Ex. "C:\Users\Laptop\Desktop\Coding\LLM\Day18\knowledge-base\employees\"  -> "employees"
            # Ex. "C:\Users\Laptop\Desktop\Coding\LLM\Day18\knowledge-base\products\"  -> "products"
            doc_type = os.path.basename(folder)
            print(f"Loading documents from: {folder} (Document type: {doc_type})")
            # Use DirectoryLoader to load all markdown files in the folder
            loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

            # Load documents from the folder
            folder_docs = loader.load()
            # For each document loaded, add metadata and append to the documents list
            for doc in folder_docs:
                # Add metadata to each document which extracted from the folder name using the LangChain's DirectoryLoader class
                # The metadata will include the source path and the document type
                # Ex. "C:\Users\Laptop\Desktop\Coding\LLM\Day18\knowledge-base\employees\"  -> "employees"
                # Ex. "C:\Users\Laptop\Desktop\Coding\LLM\Day18\knowledge-base\products\"  -> "products"
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        else:
            print(f"Skipping non-directory item: {folder}")

    return documents

# ------------------------------------ Load the Documents ----------------------------------
# Define the base directory for the knowledge base folder
knowledge_base_root = os.path.join(script_dir, "knowledge-base")

# Use glob to find all subdirectories within 'knowledge-base'
folders = glob.glob(os.path.join(knowledge_base_root, "*"))

# Load documents from the knowledge base folders
documents = load_documents_from_folder(folder_path=folders)

# Example document structure:
# [Document(metadata={'source': 'c:\\Users\\Laptop\\Desktop\\Coding\\LLM\\Day19\\knowledge-base\\company\\about.md', 'doc_type': 'company'}, 
# page_content="# About Insurellm\n\nInsurellm was founded by Avery Lancaster...."),
# ...,
# ...,]

print("\n")


# ------------------------------------ Break Down Documents into Overlapping Chunks ----------------------------------
# Create a text splitter to break down the documents into chunks.  This will break down all of the data into
# roughly 1000 character sized chunks in sensible boundaries.  The chunks can relate or overlap by 200 characters.  
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# Split the documents into chunks
# This will create overlapping chunks of text from the documents
# The chunk size is set to 1000 characters, and the overlap is set to 200 characters
chunks = text_splitter.split_documents(documents)


# ------------------------------------ Inspect the Chunks Created by LangChain ----------------------------------
print(f"Insepecting the chunks:")
print(f"*{len(chunks)} number of chunks were created")

# Print the unique document types found in the chunks using set to keep only unique values
doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"*Document types found: {', '.join(doc_types)}")


# ------------------------------------ Vector Embeddings ----------------------------------
# Create an embedding model using OpenAI's embedding API
# The langchain-openai library (specifically OpenAIEmbeddings and ChatOpenAI) automatically looks for the 
# 'OPENAI_API_KEY' environment variable. When you instantiate OpenAIEmbeddings():
embeddings = OpenAIEmbeddings()

# Define the path where the Chroma vector database will be stored
db_name = r"C:\Users\Laptop\Desktop\Coding\LLM\Day20\vector_db"

# Create a FAISS (Facebook AI Similarity Search) vector store from the pre-chunked markdown (*.md) documents.
# This will generate vector embeddings and store them in memory.  
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# Analyze the vectorstore
total_vectors = vectorstore.index.ntotal
dimensions = vectorstore.index.d

print(f"*There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store") # There are 123 vectors with 1,536 dimensions in the vector store


# ------------------------------------ Prepare Data for Visualization ----------------------------------
vectors = []    # Stores the actual vector embeddings.
documents = []  # Stores the original text content of the documents.
doc_types = []  # Stores the document types (e.g., 'products', 'employees').
colors = []     # Stores the corresponding colors for visualization, based on document type.

# These colors will be used to visually distinguish different document (*.md) types in the t-SNE plots.
color_map = {
    'products': '#A2D2DF',      # Blue
    'employees': '#A5B68D',     # Green
    'contracts': '#BC7C7C',     # Red
    'company': '#E4C087'        # Orange
}

# Iterate through each vector in the FAISS index to retrieve its embedding, original document content, and associated metadata.
for i in range(total_vectors):
    # Reconstruct the high-dimensional vector embedding from the FAISS index.
    vectors.append(vectorstore.index.reconstruct(i))
    # Get the document store ID corresponding to the current vector index.
    doc_id = vectorstore.index_to_docstore_id[i]
    # Use the document store ID to retrieve the full Document object, which contains both the page content and metadata.
    document = vectorstore.docstore.search(doc_id)
    # Append the original text content of the document to the 'documents' list.
    documents.append(document.page_content)
    # Extract the 'doc_type' from the document's metadata.
    doc_type = document.metadata['doc_type']
    # Append the extracted document type to the 'doc_types' list.
    doc_types.append(doc_type)
    # Look up the corresponding color for the document type from the 'color_map'
    # and append it to the 'colors' list.
    colors.append(color_map[doc_type])

    # Print the first vector
    if (i == 0):
        print(f"\nExample Vector Data:")            
        print(f"doc_id: {doc_id}")                  # "5e6572d9-8af9-4207-a7e4-dc6f5bc95337"
        print(f"document: {document}")              # "page_content='# Careers at Insurellm\n\nInsurellm is hiring!..."
        print(f"doc_type: {doc_type}")              # "company"
        print(f"color: {color_map[doc_type]}")      # "#E4C087"

# Convert the list of vectors into a NumPy array. This is often required
# for numerical operations and is the expected input format for t-SNE.
vectors = np.array(vectors)


# ------------------------------------ 2D Visualization Using t-SNE ----------------------------------
# t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction
# technique particularly well-suited for visualizing high-dimensional datasets.
# It aims to place data points in a lower-dimensional space (here, 2D or 3D)
# such that similar points are modeled by nearby points and dissimilar points
# are modeled by distant points. This helps in identifying clusters and relationships
# within the Chroma vector store.
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D FAISS Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

# fig.show()


# ------------------------------------ 3D Visualization Using t-SNE ----------------------------------
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D FAISS Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

# fig.show()


# ------------------------------------ LLM Integration with RAG ----------------------------------
# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# Set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# The retriever is an abstraction over the VectorStore that will be used during RAG
# 'vectorstore.as_retriever()' converts the vector store (e.g., FAISS index) into a retriever object.
# A retriever's role is to fetch relevant chunks of information from the vector store
# that are semantically similar to the user's query. This is the "Retrieval" part of RAG.
retriever = vectorstore.as_retriever()

# Putting it together: set up the conversation chain with the LLM, the vector store, and memory
# 'ConversationalRetrievalChain.from_llm' creates a powerful chain that integrates
# the LLM, the retriever, and the memory.
# When a query comes in:
# 1. The query, along with chat history, is sent to the LLM (or a prompt engineering step)
#    to formulate a search query.
# 2. This search query is then passed to the 'retriever' to find relevant documents
#    (chunks) from the 'vectorstore'.
# 3. The retrieved documents are then added as context to the original user query.
# 4. Finally, the LLM processes this combined information (query + context + chat history)
#    to generate a more informed and accurate answer.
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

query = "Can you describe Insurellm in a few sentences"
result = conversation_chain.invoke({"question":query})
print(result["answer"])

# 'ConversationBufferMemory' stores past messages (both user and AI) in a buffer.
# This is essential for maintaining context in a multi-turn conversation, allowing the LLM
# to understand and respond based on previous interactions.
# 'memory_key='chat_history'' specifies the key under which the chat history will be stored.
# 'return_messages=True' ensures the history is returned as a list of message objects.
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# ------------------------------------ Gradio UI ----------------------------------
def chat(message, history):
    # The 'history' parameter is ignored completely because LangChain already handles the
    # history for us.  
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat).launch()

# "Can you describe Insurellm in a few sentences."
# "What can you tell me about Emily Tran?"
# "And how was she rated in 2022?"
# "What can you tell me about the contract for EverGuard Insurance?"
