# This script demonstrates a **Retrieval-Augmented Generation (RAG)** system
# that highlights key architectural decisions and common debugging patterns in LangChain.
#
# Day 21 Recap & Day 22 Enhancements:
# In Day 21, the RAG system was set up, but it struggled with specific questions like:
# "Who received the prestigious IIOTY award in 2023?"
# This often happens when the "retrieval" part of RAG isn't providing the
# most relevant **chunks** of information to the Large Language Model (LLM).
# Day 22 focuses on diagnosing and fixing this by understanding LangChain's internals,
# particularly how **callbacks** can help, and how adjusting **chunking strategies** impacts retrieval.

# ---
# ## Demystifying LangChain & Diagnosing Retrieval Issues
#
# LangChain, at its core, orchestrates various components (LLMs, vector stores, retrievers, memory)
# into a coherent **chain**. When a RAG system fails to answer a question, the first place to look
# is usually the **retrieval** step:
#
# 1.  **Are the right chunks being retrieved?**
#     This is the most common culprit. If the vector store isn't returning documents
#     that contain the answer, the LLM, no matter how powerful, can't "know" it.
#     The initial setup in Day 21 with a default retriever likely returned a limited
#     number of chunks (often `k=4` by default in many LangChain retrievers).
#     If the relevant information was spread across many chunks or less relevant ones
#     were prioritized, the LLM wouldn't receive the complete picture.
#
# ---
# ## Using Callbacks for Diagnosis
#
# LangChain provides **Callbacks** (like `StdOutCallbackHandler`) to peek into the
# execution flow of a chain. By adding `callbacks=[StdOutCallbackHandler()]` to your
# `ConversationalRetrievalChain`, you can see:
# * What search query the LLM is formulating.
# * Which documents/chunks the retriever is fetching *before* they're passed to the LLM.
# * The final prompt sent to the LLM.
#
# This allows you to *diagnose* if the retrieval step is indeed the bottleneck. If the
# retrieved chunks don't contain the answer to "Who received the prestigious IIOTY award in 2023?",
# then you've pinpointed the problem.
#
# ---
# ## Fixing the Problem: Adjusting Chunking Strategies (The Retriever)
#
# Once diagnosed, the fix often involves improving the retrieval mechanism. This script specifically
# demonstrates adjusting the `k` parameter for the **retriever**:
#
# * **`retriever = vectorstore.as_retriever(search_kwargs={"k": 25})`**:
#     Here, `k=25` means the retriever will fetch the **top 25 most semantically similar chunks**
#     from the `vectorstore` based on the user's query. In Day 21, without explicitly setting `k`,
#     LangChain's default (often a small number like 4) was used.
#     By increasing `k`, you provide the LLM with a much broader context, significantly
#     increasing the chances that the relevant information is included. This is precisely
#     why the question about the "IIOTY award" was answerable after this change.
#
#     **Does increasing `k` send more data?** Each "chunk" is a piece of text.
#     Fetching `25` chunks sends significantly more textual data as context to the LLM
#     than fetching `4` chunks. While this improves accuracy by providing more context,
#     it also means:
#     * **Higher token usage:** More data means more tokens, potentially leading to higher API costs.
#     * **Longer processing times:** LLMs take longer to process larger contexts.
#     * **Context window limits:** LLMs have a maximum context window. If `k` is too high,
#         you might exceed this limit, causing errors or truncation.
#
# In essence, **LangChain works by carefully coordinating these steps**: a query comes in,
# a retriever pulls relevant context, memory maintains conversation flow, and the LLM synthesizes
# a response. Understanding and tuning each piece, especially the retriever's chunking strategy,
# is crucial for a robust RAG system.


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
from langchain_core.callbacks import StdOutCallbackHandler
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
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

# The LLM fails to find who won the award 
query = "Who received the prestigious IIOTY award in 2023?" 
result = conversation_chain.invoke({"question":query})
print(result["answer"]) # Response: I don't know.


# ------------------------------------ Using callbacks to look under the hood at was sent to the LLM to diagnose why I couldn't answer the question ----------------------------------
# > Entering new ConversationalRetrievalChain chain...


# > Entering new StuffDocumentsChain chain...


# > Entering new LLMChain chain...
# Prompt after formatting:
# System: Use the following pieces of context to answer the user's question. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# ----------------
# - **2022**: **Satisfactory**  
#   Avery focused on rebuilding team dynamics and addressing employee concerns, leading to overall improvement despite a saturated market.  

# - **2023**: **Exceeds Expectations**  
#   Market leadership was regained with innovative approaches to personalized insurance solutions. Avery is now recognized in industry publications as a leading voice in Insurance Tech innovation.

# ## Annual Performance History
# - **2020:**  
#   - Completed onboarding successfully.  
#   - Met expectations in delivering project milestones.  
#   - Received positive feedback from the team leads.

# - **2021:**  
#   - Achieved a 95% success rate in project delivery timelines.  
#   - Awarded "Rising Star" at the annual company gala for outstanding contributions.  

# - **2022:**  
#   - Exceeded goals by optimizing existing backend code, improving system performance by 25%.  
#   - Conducted training sessions for junior developers, fostering knowledge sharing.  

# - **2023:**  
#   - Led a major overhaul of the API internal architecture, enhancing security protocols.  
#   - Contributed to the company’s transition to a cloud-based infrastructure.  
#   - Received an overall performance rating of 4.8/5.

# ## Annual Performance History
# - **2018**: **3/5** - Adaptable team player but still learning to take initiative.
# - **2019**: **4/5** - Demonstrated strong problem-solving skills, outstanding contribution on the claims project.
# - **2020**: **2/5** - Struggled with time management; fell behind on deadlines during a high-traffic release period.
# - **2021**: **4/5** - Made a significant turnaround with organized work habits and successful project management.
# - **2022**: **5/5** - Exceptional performance during the "Innovate" initiative, showcasing leadership and creativity.
# - **2023**: **3/5** - Maintaining steady work; expectations for innovation not fully met, leading to discussions about goals.

# ## Annual Performance History
# - **2023:** Rating: 4.5/5  
#   *Samuel exceeded expectations, successfully leading a cross-departmental project on AI-driven underwriting processes.*

# - **2022:** Rating: 3.0/5  
#   *Some challenges in meeting deadlines and collaboration with the engineering team. Received constructive feedback and participated in a team communication workshop.*

# - **2021:** Rating: 4.0/5  
#   *There was notable improvement in performance. Worked to enhance model accuracy, leading to improved risk assessment outcomes for B2C customers.*

# - **2020:** Rating: 3.5/5  
#   *Exhibited a solid performance during the initial year as a Senior Data Scientist but had struggles adapting to new leadership expectations.*

# ## Compensation History
# - **2023:** Base Salary: $115,000 + Bonus: $15,000  
#   *Annual bonus based on successful project completions and performance metrics.*
# Human: Who received the prestigious IIOTY award in 2023?

# > Finished chain.

# > Finished chain.

# > Finished chain.
# I don't know.

# ------------------------------------ Modifying the Relevant Chunks Sent ----------------------------------
# 'ConversationBufferMemory' stores past messages (both user and AI) in a buffer.
# This is essential for maintaining context in a multi-turn conversation, allowing the LLM
# to understand and respond based on previous interactions.
# 'memory_key='chat_history'' specifies the key under which the chat history will be stored.
# 'return_messages=True' ensures the history is returned as a list of message objects.
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Initialize the retriever. The retriever is a component responsible for fetching relevant information (chunks)
# from the VectorStore during a Retrieval Augmented Generation (RAG) process.
# 'search_kwargs' is a dictionary of keyword arguments passed to the search method of the retriever.
# 'k' specifies the number of top-k most relevant document chunks to retrieve. 
# As you increase the chunks to send, you send more data so that the LLM can be better trained. 
# This allows the LLM to answer this question, 'Who received the prestigious IIOTY award in 2023?'
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

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
# "Who received the prestigious IIOTY award in 2023?"