# ------------------------------------ LangChain Package ----------------------------------
# 1.    Read in the documents from the knowledge base
#       "C:\Users\Laptop\Desktop\Coding\LLM\Day18\knowledge-base\"
# 2.    Add meta-data to the documents using LangChain package
# 3.    Break down the documents into overlapping chunks
# 4.    Iterate through each chunk to search for relevant data 


# ------------------------------------ Chunking in LangChain (and RAG) ------------------------------------
# --- What is Chunking? ---
# Chunking is the process of breaking down large pieces of text (like documents, articles, PDFs,
# or entire books) into smaller, more manageable segments called "chunks."
# These chunks are typically designed to be semantically coherent and of a specific size,
# suitable for processing by Large Language Models (LLMs) and retrieval systems.

# --- Why is Chunking Essential for LLMs and RAG? ---
# 1. LLM Context Window Limitations:
#    - LLMs have a strict limit on the amount of text they can process in a single input (their "context window").
#    - Large documents often exceed this limit. Chunking ensures that the retrieved pieces of
#      information will fit within the LLM's capacity.
# 2. Improved Retrieval Relevance (Semantic Search):
#    - When you convert text into numerical "embeddings" for semantic search, more focused
#      chunks lead to more accurate embeddings.
#    - A query is more likely to find a precise, relevant chunk than to correctly identify
#      a small piece of information within a massive, undifferentiated document.
#    - This improves the "signal-to-noise" ratio, feeding the LLM only the most pertinent information.
# 3. Reduced Hallucinations & Enhanced Factuality:
#    - By providing the LLM with specific, retrieved chunks, you "ground" its responses in verifiable
#      external data, significantly reducing the tendency to invent information.
# 4. Cost and Performance Optimization:
#    - Sending smaller chunks to LLMs (especially via APIs) is typically faster and more cost-effective
#      as API pricing is often based on token usage.

# --- How LangChain Facilitates Chunking ---
# LangChain provides a robust set of `TextSplitter` classes to perform chunking.
# These splitters offer various strategies to intelligently divide text while trying to maintain
# semantic coherence and context.

# Key Text Splitter Classes in LangChain:

# 1. `RecursiveCharacterTextSplitter`:
#    - **Most commonly used.** It attempts to split text using a list of characters (e.g., ["\n\n", "\n", " ", ""]).
#    - It tries to split on the largest delimiter first. If the chunk is still too big, it tries the next smaller one.
#    - This recursive approach helps preserve logical sections (paragraphs, sentences).
#    - Key parameters: `chunk_size` (maximum size of each chunk) and `chunk_overlap` (how much
#      text to share between consecutive chunks to maintain context across splits).

# 2. `CharacterTextSplitter`:
#    - A simpler splitter that divides text based on a single character (default is "\n\n").

# 3. `MarkdownTextSplitter` / `HTMLTextSplitter`:
#    - Specialized splitters that understand the structure of specific file types (Markdown, HTML).
#    - They aim to split content in a way that respects the document's inherent hierarchy (e.g., not splitting in the middle of a heading or code block).

# 4. `SemanticChunker` (from `langchain_experimental`):
#    - An advanced splitter that tries to split based on semantic meaning, often by embedding
#      sentences and looking for "breaks" in semantic similarity.

# --- Common Chunking Parameters: ---
# - `chunk_size`: The desired maximum length of each text chunk (in characters or tokens).
# - `chunk_overlap`: The number of characters or tokens that adjacent chunks will share.
#   - **Purpose of Overlap:** It helps ensure that context isn't lost at the boundaries
#     between chunks. If the answer to a query spans two chunks, overlap increases the
#     chance that all necessary information will be present in at least one retrieved chunk.

# --- Typical Workflow with Chunking in LangChain RAG: ---
# 1. Load documents using a `DocumentLoader` (e.g., `PyPDFLoader`, `WebBaseLoader`).
# 2. Split documents into chunks using a `TextSplitter` (e.g., `RecursiveCharacterTextSplitter`).
# 3. Create embeddings for each chunk using an `Embeddings` model (e.g., `OpenAIEmbeddings`).
# 4. Store the chunks and their embeddings in a `VectorStore` (e.g., `Chroma`, `Pinecone`).
# 5. When a query comes in, embed the query and perform a similarity search in the `VectorStore`
#    to retrieve the most relevant chunks.
# 6. Pass these retrieved chunks as context to the LLM along with the user's original query.


# ------------------------------------ Packages ----------------------------------
# pip install anthropic
# pip install dotenv
# pip install gradio
# pip install langchain
# pip install -U langchain-community
# pip install openai
# pip install python-dotenv


# ------------------------------------ Imports ----------------------------------
import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter


# ------------------------------------ Constants / Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))


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

print(f"\nLoaded {len(documents)} documents from the knowledge base.\n")

# Iterate through one document to print its metadata and content
print("\n ------- Document Metadata Example: -------\n")
for doc in documents[:1]:
    # Print document metadata
    print(f"Document Metadata: {doc.metadata}")
    # Print document source path
    print(f"Document Source: {doc.metadata['source']}")
    # Print the document type
    print(f"Document Type: {doc.metadata['doc_type']}")
    # Print first 50 characters of content
    print(f"Document Content: {doc.page_content[:50]}...")  

print("\n--------------------------------------------\n")

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
# Print the chunk length
print("*Example Chunk Metadata and Content:")
for chunk in chunks[:1]:
    # Print chunk metadata
    print(f"      -Chunk Metadata: {chunk.metadata}")
    # Print first 50 characters of chunk content
    print(f"      -Chunk Content: {chunk.page_content[:50]}...")
# print("\n---------------------------------------------\n")

# Print the unique document types found in the chunks using set to keep only unique values
doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"*Document types found: {', '.join(doc_types)}")


# ------------------------------------ Keyword Search ----------------------------------
# Iterate through each chunk to look for 'CEO' keyword
ceo_chunk_count = 0

print(f"*Chunks containing 'CEO':")
for i in range(0, len(chunks)):
    chunk = chunks[i]
    if 'ceo' in chunk.page_content.lower():
        print(f"\n-------------- Chunk {i} contains 'CEO' -------------- \n")
        print(chunk)
        # Add to the count of chunks containing 'CEO'
        ceo_chunk_count +=1

print(f"\n\n{ceo_chunk_count} number of chunks contain the keyword, 'CEO' (not case sensitive).")


# ------------------------------------ Output ----------------------------------
"""
Loading documents from: c:\Users\Laptop\Desktop\Coding\LLM\Day19\knowledge-base\company (Document type: company)
Loading documents from: c:\Users\Laptop\Desktop\Coding\LLM\Day19\knowledge-base\contracts (Document type: contracts)
Loading documents from: c:\Users\Laptop\Desktop\Coding\LLM\Day19\knowledge-base\employees (Document type: employees)
Loading documents from: c:\Users\Laptop\Desktop\Coding\LLM\Day19\knowledge-base\products (Document type: products)

Loaded 31 documents from the knowledge base.


 ------- Document Metadata Example: -------

Document Metadata: {'source': 'c:\\Users\\Laptop\\Desktop\\Coding\\LLM\\Day19\\knowledge-base\\company\\about.md', 'doc_type': 'company'}
Document Source: c:\Users\Laptop\Desktop\Coding\LLM\Day19\knowledge-base\company\about.md
Document Type: company
Document Content: # About Insurellm

Insurellm was founded by Avery ...

--------------------------------------------

Created a chunk of size 1088, which is longer than the specified 1000
Insepecting the chunks:
*123 number of chunks were created
*Example Chunk Metadata and Content:
      -Chunk Metadata: {'source': 'c:\\Users\\Laptop\\Desktop\\Coding\\LLM\\Day19\\knowledge-base\\company\\about.md', 'doc_type': 'company'}
      -Chunk Content: # About Insurellm

Insurellm was founded by Avery ...
*Document types found: company, contracts, employees, products
*Chunks containing 'CEO':

-------------- Chunk 35 contains 'CEO' --------------

page_content='3. **Regular Updates:** Insurellm will offer ongoing updates and enhancements to the Homellm platform, including new features and security improvements.

4. **Feedback Implementation:** Insurellm will actively solicit feedback from GreenValley Insurance to ensure Homellm continues to meet their evolving needs.

---

**Signatures:**

_________________________________
**[Name]**
**Title**: CEO
**Insurellm, Inc.**

_________________________________
**[Name]**
**Title**: COO
**GreenValley Insurance, LLC**

---

This agreement represents the complete understanding of both parties regarding the use of the Homellm product and supersedes any prior agreements or communications.' metadata={'source': 'c:\\Users\\Laptop\\Desktop\\Coding\\LLM\\Day19\\knowledge-base\\contracts\\Contract with GreenValley Insurance for Homellm.md', 'doc_type': 'contracts'}

-------------- Chunk 54 contains 'CEO' --------------

page_content='## Support

1. **Customer Support**: Velocity Auto Solutions will have access to Insurellm’s customer support team via email or chatbot, available 24/7.
2. **Technical Maintenance**: Regular maintenance and updates to the Carllm platform will be conducted by Insurellm, with any downtime communicated in advance.
3. **Training & Resources**: Initial training sessions will be provided for Velocity Auto Solutions’ staff to ensure effective use of the Carllm suite. Regular resources and documentation will be made available online.

---

**Accepted and Agreed:**
**For Velocity Auto Solutions**
Signature: _____________________
Name: John Doe
Title: CEO
Date: _____________________

**For Insurellm**
Signature: _____________________
Name: Jane Smith
Title: VP of Sales
Date: _____________________' metadata={'source': 'c:\\Users\\Laptop\\Desktop\\Coding\\LLM\\Day19\\knowledge-base\\contracts\\Contract with Velocity Auto Solutions for Carllm.md', 'doc_type': 'contracts'}

-------------- Chunk 65 contains 'CEO' --------------

page_content='# Avery Lancaster

## Summary
- **Date of Birth**: March 15, 1985
- **Job Title**: Co-Founder & Chief Executive Officer (CEO)
- **Location**: San Francisco, California

## Insurellm Career Progression
- **2015 - Present**: Co-Founder & CEO
  Avery Lancaster co-founded Insurellm in 2015 and has since guided the company to its current position as a leading Insurance Tech provider. Avery is known for her innovative leadership strategies and risk management expertise that have catapulted the company into the mainstream insurance market.

- **2013 - 2015**: Senior Product Manager at Innovate Insurance Solutions
  Before launching Insurellm, Avery was a leading Senior Product Manager at Innovate Insurance Solutions, where she developed groundbreaking insurance products aimed at the tech sector.' metadata={'source': 'c:\\Users\\Laptop\\Desktop\\Coding\\LLM\\Day19\\knowledge-base\\employees\\Avery Lancaster.md', 'doc_type': 'employees'}


3 number of chunks contain the keyword, 'CEO' (not case sensitive).
"""