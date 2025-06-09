# ------------------------------------ Vector Embedding Models ------------------------------------
# Vector embeddings are numerical representations of text data in high-dimensional space.
# Each text chunk (sentence, paragraph, doc) is converted into a fixed-size vector
# using a pre-trained embedding model (like OpenAI's `text-embedding-ada-002`).
#
# The key properties of these embeddings:
#   • Similar meanings → vectors close together (small cosine distance)
#   • Dissimilar meanings → vectors far apart
#
# 🔍 WHY USE VECTOR EMBEDDINGS?
#   • Enables semantic search → retrieve text that's *meaningfully similar*, not just matching keywords.
#   • Foundation for Retrieval-Augmented Generation (RAG) pipelines in LLM apps.
#   • Useful for clustering, classification, recommendation, and visualization of text data.
#
# 🧠 HOW THEY WORK IN THIS PROJECT:
#   • Documents are split into overlapping chunks (~1000 characters).
#   • Each chunk is embedded using OpenAI's embedding model.
#   • The embeddings are stored in a persistent vector database (For this script, Chroma).
#   • Later, we can query this DB semantically or visualize it using t-SNE.
#
# 🛠️ ANALOGY:
#   • Think of embeddings as putting text into a "map of meaning."
#   • Similar ideas are grouped geographically, making it easy to find neighbors.
#
# 💡 FURTHER CONSIDERATIONS FOR EMBEDDINGS:
#   • **Model Choice**: While OpenAI's `text-embedding-ada-002` is popular, other models like
#     Cohere, Google's `text-embedding-gecko`, or open-source models (e.g., from Hugging Face
#     like `sentence-transformers`) might offer different performance characteristics,
#     cost, or privacy benefits. The choice often depends on the specific use case,
#     data domain, and inference speed requirements.
#   • **Embedding Dimensions**: Different models produce embeddings of varying dimensions.
#     `text-embedding-ada-002` yields 1536 dimensions, which is relatively high-dimensional.
#     While more dimensions can capture richer semantic nuances, they also increase
#     storage and computational cost.
#   • **Updating Embeddings**: For dynamic datasets, strategies for updating the vector
#     store (e.g., incremental updates, re-embedding changed documents, handling deletions)
#     are crucial to keep the semantic search results fresh and accurate.
#   • **Quantization**: For very large-scale applications, techniques like
#     quantization (e.g., Product Quantization, Locality Sensitive Hashing) can reduce
#     the memory footprint and speed up similarity search by compressing embeddings,
#     though often with a slight trade-off in accuracy.


# ------------------------------------ Packages ----------------------------------
# pip install anthropic
# pip install chromadb
# pip install dotenv
# pip install gradio
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
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


# ------------------------------------ Constants / Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
pio.renderers.default = 'browser'  # Or try 'chrome', 'firefox' explicitly


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

# If a vector store already exists at this path, delete the existing collection to start fresh
# This deletes the Chroma database in "C:\Users\Laptop\Desktop\Coding\LLM\Day20\vector_db\"
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, 
           embedding_function=embeddings).delete_collection()

# Create a Chroma vector store from the pre-chunked markdown (*.md) documents.  This will generate vector 
# embeddings and store them on disk.  Basically, the embeddings are stored in a persistent vector db (Chroma)
vectorstore = Chroma.from_documents(documents=chunks, 
                                    embedding=embeddings, 
                                    persist_directory=db_name)
print(f"*Vectorstore created with {vectorstore._collection.count()} documents")  # Vectorstore created with 123 documents


# ------------------------------------ Inspect Embedding Dimensions ----------------------------------
# Access the raw underlying collection of vectors
collection = vectorstore._collection

# Retrieve a single sample embedding to inspect its dimensionality
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"*The vectors have {dimensions:,} dimensions for one sample chunk.")    # The vectors have 1,536 dimensions


# ------------------------------------ Prepare Data for Visualization ----------------------------------
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
metadatas = result['metadatas']

# Assign colors to doc_types
type_color_map = {
    'products': '#A2D2DF',          # Blue
    'employees': '#A5B68D',         # Green
    'contracts': '#BC7C7C',         # Red
    'company': '#E4C087'            # Orange
}
colors = [type_color_map.get(t, 'gray') for t in doc_types]

# Create a hover text with doc_type, source path, and preview
hover_text = [
    f"<b>Type:</b> {t}<br>"
    f"<b>Path:</b> {m.get('source', 'N/A')}<br>"
    f"<b>Preview:</b> {d[:50].replace('<', '&lt;').replace('>', '&gt;')}..."
    for t, d, m in zip(doc_types, documents, metadatas)
]


# ------------------------------------ 2D Visualization Using t-SNE ----------------------------------
# t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction
# technique particularly well-suited for visualizing high-dimensional datasets.
# It aims to place data points in a lower-dimensional space (here, 2D or 3D)
# such that similar points are modeled by nearby points and dissimilar points
# are modeled by distant points. This helps in identifying clusters and relationships
# within the Chroma vector store.

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=hover_text,
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    xaxis_title='t-SNE Dim 1',
    yaxis_title='t-SNE Dim 2',
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)
fig.show()


# ------------------------------------ 3D Visualization Using t-SNE ----------------------------------
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=hover_text,
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='t-SNE Dim 1', yaxis_title='t-SNE Dim 2', zaxis_title='t-SNE Dim 3'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)
fig.show()


# ------------------------------------ Enhanced 5D 3D t-SNE Visualization ----------------------------------

# from sklearn.manifold import TSNE
# import plotly.graph_objects as go

# Reduce vectors to 3D with t-SNE
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Calculate chunk lengths to scale marker sizes (min=3, max=10 for visibility)
chunk_lengths = [len(doc) for doc in documents]
marker_sizes = [max(3, min(10, l / 100)) for l in chunk_lengths]  # Scale size between 3–10

# Hover text: show document type and snippet of text
hover_text = [f"Type: {t}<br>Length: {len(d)}<br>Text: {d[:100].replace('<', '&lt;').replace('>', '&gt;')}..." 
              for t, d in zip(doc_types, documents)]

# Assign colors to doc_types
type_color_map = {
    'products': '#A2D2DF',          # Blue
    'employees': '#A5B68D',         # Green
    'contracts': '#BC7C7C',         # Red
    'company': '#E4C087'            # Orange
}
colors = [type_color_map.get(t, 'gray') for t in doc_types]

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(
        size=marker_sizes,      # Size encodes chunk length
        color=colors,           # Color encodes doc type
        opacity=0.8,
    ),
    text=hover_text,            # Custom hover text
    hoverinfo='text'
)])

# Layout styling
fig.update_layout(
    title='5D Embedding Visualization (3D t-SNE + Color + Size)',
    scene=dict(
        xaxis_title='t-SNE X',
        yaxis_title='t-SNE Y',
        zaxis_title='t-SNE Z'
    ),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

# Show the plot
fig.show()
