# 🔥 **LLM Engineering Playground** 🔥

### Exploring the Frontier of AI & Crafting Unique Solutions

This repository chronicles my hands-on adventure into Large Language Model (LLM) engineering. While I'm coding along with an online course, I'm deliberately diverging to infuse my own creative twists and explore practical, real-world applications. Expect to see custom integrations, unique AI interactions, and a focus on getting my hands dirty with powerful language models.


## 🚀 **My LLM Explorations (By Day)**

Here's a peek into the evolving projects and concepts within this repository:

---
### **Day 01: Cloud-Powered Conversations & Summaries** 🗪 
- **Objective:** Harnessing the power of cloud-based LLMs for basic chat and content summarization.
- **Tech Stack:** [OpenAI API Platform](https://platform.openai.com/docs/api-reference/introduction) (using 'gpt-4o-mini').
- **Highlights:** Built a system to automate interactions with language models and extract key information from scraped websites. Imagine instantly summarizing an entire article!
---
### **Day 02: Configuring AI Chatbot Locally with Ollama** 🦙 
- **Objective:** Enabling local execution of an AI model (LLaMA) for offline capabilities.
- **Tech Stack:** Ollama API, LLaMA model.
- **Highlights:** Successfully set up a local LLM environment, proving that powerful AI isn't always cloud-bound. This opens up possibilities for privacy-focused or resource-constrained applications.
---
### **Day 05: AI-Powered Portfolio Generation** 🌐 
- **Objective:** Leveraging AI to dynamically create a markdown portfolio of my GitHub projects.
- **Tech Stack:** BeautifulSoup (for web scraping), OpenAI's ['gpt-4o-mini'](https://platform.openai.com/docs/models/gpt-4o-mini) API.
- **Highlights:** Developed a script that scrapes my GitHub for relevant project links and then uses an LLM to automatically generate a well-formatted markdown portfolio. Automating content creation for showcasing work!
---
### **Day 06: Configuring different LLM platforms to engage in a defined conversation together** 🎭
- **Objective:** Configuring different LLM platforms to adopt contrasting personas and engage in a defined conversation.  I configure GPT to be a gullable child and Gemini to be a teacher that loves telling elaborate lies involving Dragons and Unicorns.  I make these AI platforms talk to one another.  
- **Tech Stack:** OpenAI's 'GPT-4o', Google's Gemini 1.5 Flash.
- **Highlights:** Witness a hilarious dialogue between a "gullible child" GPT and a "teacher who loves telling elaborate lies involving Dragons and Unicorns" Gemini. Overcame GPT's statelessness by maintaining conversation history. The entire dialog is saved to an HTML file for easy viewing!
    - 🔗 [View the conversation HTML here!](https://html.onlineviewer.net/) (Paste the generated HTML content here for easier viewing)
---
### **Day 07: Gradio Web App - AI Photo Editor** ✨
- **Objective:** Creating a simple web application for AI-powered photo editing.
- **Tech Stack:** Gradio, OpenAI's 'dall-e-2' model.
- **Highlights:** Built an interactive web app where users can input text prompts to dynamically edit images using DALL-E 2.
    - ![Gradio Web Server in Action](https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day07/Gradio%20Web%20Server.png)
---
### **Day 08: Shiny App Chatbot - Your AI Art Idea Assistant** 🎨
- **Objective:** Developing an AI Chatbot as a Shiny App, focusing on maintaining conversational context.
- **Tech Stack:** OpenAI API, Shiny App.
- **Highlights:** Created an "AI Art Idea Generator Assistant" chatbot specifically designed to help users brainstorm creative ideas for pet-themed AI-generated images by configuring system context. It suggests art styles, colors, and themes!
    - **Example Chat with the OpenAI Chatbot:**
        ![Shiny App Chatbot Interaction](https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day08/AI%20Prompt.jpg)
    - **Putting the Idea into Action (using ChatGPT for Image Generation with the idea):**
        <p align="center">
            <img src="https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day08/ChatGPT%20Prompt.jpg" alt="ChatGPT generated image prompt" width="50%" height="auto">
        </p>
    - **And the Amazing Result!** (Cats with a Salvador Dalí influence!)
        <p align="center">
            <img src="https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day08/Cats%20with%20a%20Salvador%20Dali%20Influence.jpg" alt="AI-generated cats with surreal influence" width="40%" height="auto">
        </p>
---
### **Day 09: Advanced LLM Tooling** 🛠️
- **Objective:** Developing an AI Chatbot as a Gradio App, focusing on maintaining conversational context and equipping the bot with a custom tool to add additional knowledge to the bot.  This bot provides nutritional data for fruits from a specific source (through API integration) and alerts the end user if it is unable to find data through the API call.  I used a free fruit data API that doesn't require authentication just to get the code up & running faster.  But for proof of concept, integrating the chat bot with API interactions is a very powerful tool because the data source can be controlled or the LLM can be trained with additional data.  
- **Tech Stack:** OpenAI API, Gradio App, API integration (Fruitvice API)
- **Highlights:** Robust API integration gracefully handling when a fruit is not found in the API call.
- **Example Chat with the OpenAI Chatbot:**
        ![Gradio App Chatbot Interaction](https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day09/Gradio%20Fruit%20Chatbot.jpg)
---
### **Day 10: Multimodal AI Assistant, Image & Sound Generation** 🤖
- **Objective:** Build an interactive multimodal (combine natural language understanding w/image generation & speech synthesis) AI assistant that can
    - 💬 Chat with users using OpenAI's GPT model.
    - 🎨 Generate images based on user prompts via DALL·E 3.
    - 🔊 Speak its responses aloud using text-to-speech (TTS).
    - 🖥️ Present everything in a user-friendly web interface using Gradio.
- **Tech Stack:** OpenAI API (Dall-e-3 for image generation), Gradio App, tts-1: Text-to-speech 
- **Highlights:** Seamless conversation flow powered by GPT.  Real-time image generation from user descriptions
    - **Example Chat with the OpenAI Chatbot:**
        ![Shiny App Chatbot Interaction](https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day10/Dashboard.jpg)
    - **This is one of the photos that I had my chatbot create.  I think this is one the coolest things.**
        <p align="center">
            <img src="https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day10/Downloads/66870a9294ff47bc9cc0636f0aba180d.jpg" alt="ChatGPT generated image prompt" width="40%" height="auto">
        </p>
---
### **Day 11: Surrealistic Image Generation with FluxPipeline on GPU/CPU** 🎨  
- **Objective:** Generate complex, surrealistic images using diffusion-based models from Hugging Face with efficient GPU acceleration or fallback to CPU.  
    - 🔥 Load and run the FluxPipeline model for text-to-image generation with precision control and reproducibility via manual seeding.  
    - 🌌 Create vivid, imaginative scenes with detailed textual prompts.  
    - 💻 Provide GPU info to verify hardware acceleration availability (commented for optional use).  
- **Tech Stack:** Hugging Face `diffusers` FluxPipeline, PyTorch (CUDA), Google Colab userdata API for secure key management, PIL for image handling.  
- **Highlights:**  
    - Flexible runtime support (GPU with float16 precision or CPU fallback).  
    - Use of seeded random generators for consistent image outputs.  
    - Elaborate prompt engineering to create highly detailed surreal imagery.  
    - Integration with Hugging Face Hub authentication via token 🔑 login.
---
### **Day 12: Hugging Face Pipelines & Model Integration with Google Colab** 🚀  
- **Objective:** Integrate multiple Hugging Face pipelines in Google Colab for various NLP and multimodal tasks such as sentiment analysis, translation, image generation, and text-to-speech.  
    - 🧠 Leverage GPU acceleration for fast processing of models like sentiment analysis, text summarization, and translation.
    - 🎨 Generate images using the Stable Diffusion model and synthesize speech from text with Microsoft's TTS model.
- **Tech Stack:** 
    - Hugging Face Transformers & Diffusers
    - Google Colab for API management and hosting
    - PyTorch (CUDA) for GPU acceleration
- **Highlights:**  
    - Integration with Hugging Face models for diverse NLP and multimodal tasks.
    - Image generation and text-to-speech synthesis with pre-trained models.
---
### **Day 13: Tokenizers - Exploring LLM Inputs** 🧠 
- **Objective:** Understand how LLMs process text by encoding and decoding with various tokenizers.
- **Tech Stack:** 
    - Hugging Face Transformers, Colab, Meta’s LLaMA 3.1, Phi-3, Qwen2, Starcoder2.
- **Highlights:**  
    - Used Hugging Face open-source models to tokenize text and code, applied chat templates, and compared token outputs across multiple models.
---
### **Day 14: Under the Hood of Transformers** 🔧
- **Objective:** Demystify the internals of modern Transformer models by working directly with their lower-level PyTorch-based APIs using the Hugging Face transformers library. Use memory-efficient 4-bit quantization.  
- **Tech Stack:** 
    - Google Colab (Free T4 GPU runtime)
    - Hugging Face Transformers
    - PyTorch
    - BitsAndBytes (for 4-bit quantization)
- **Highlights:**  
    - This notebook provides a hands-on tour of five state-of-the-art open-source Transformer models:
        - 🦙 **LLaMA 3.1** by Meta
        - 🧠 **Phi-3** by Microsoft
        - 💎 **Gemma** by Google
        - 🌪️ **Mixtral** by Mistral
        - 🐉 **Qwen** by Alibaba Cloud
---
### **Day 15: Audio to Text Summarization** 🎧
- **Objective:** This notebook transforms audio into structured meeting minutes using advanced AI.  
- **Tech Stack:** 
    - 🤫 **Whisper-1** by OpenAI (For speech-to-text transcription)
    - 🦙 **LLaMA 3.1** by Meta (For summarizing text & extract key details)
    - BitsAndBytes (for 4-bit quantization)
- **Highlights:**
    - Audio transcription, meeting minutes generation, quantization for efficiency, streamed output (demonstrate how to stream LLM's output for a more interactive human-like experience), & Google Drive integration.  
---
### **Day 16: Code Alchemist — Python to Lightning-Fast Languages** ⚡
- **Objective:** Transform Python scripts into high-performance equivalents in languages like C++, Rust, Go, and more using state-of-the-art LLMs.
- **Tech Stack:** 
  - [OpenAI API](https://platform.openai.com/docs/api-reference/introduction)
  - [Anthropic Claude API](https://console.anthropic.com/)
- **Highlights:**  
  - Converts Python into various target languages with a focus on **performance** and **output fidelity**  
  - Cleans up LLM responses (e.g. removes markdown artifacts, formats extensions)  
---
### **Day 17: Code Alchemist++ — Multi-LLM, Multi-Language Optimization** 🧪⚙️
- **Objective:** Expand Day 16's converter into a multi-model, multi-language code optimizer by integrating Hugging Face inference endpoint.
- **New Capabilities:** 
  - [CodeQwen1.5-7B-Chat](https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat)
---
### **Day 18: RAG System - Contextualized LLM Chatbot** 📚
- **Objective:** Develop a Retrieval Augmented Generation (RAG) system to enhance an LLM's ability to answer questions by providing access to an external, domain-specific knowledge base. 
- **Tech Stack:**
    - OpenAI API ('gpt-4o-mini')
- **Highlights:**
    - **RAG Implementation:** Built a system that dynamically retrieves relevant context (employee and product information) from a local knowledge base (`.md` files) and injects it into the LLM's prompt, reducing hallucinations and providing up-to-date information.
    - **Knowledge Base Integration:** The system loads markdown files from specified directories (`knowledge-base/employees` and `knowledge-base/products`) into a `context` dictionary, making the information readily available for retrieval.
    - **Context Retrieval Logic:** Implemented a `get_relevant_context` function that intelligently identifies and extracts pertinent information based on keywords in the user's query.
    - **Prompt Augmentation:** The `add_context` function seamlessly appends the retrieved context to the user's message before sending it to the LLM, ensuring the model has the necessary information to formulate accurate responses.
    - **Gradio Chat Interface:** A user-friendly Gradio interface allows for interactive conversations, demonstrating the RAG system in action.
- **How it Works:**
    -  **Ingestion (Pre-processing):** Markdown files containing employee profiles and product details are loaded into a Python dictionary, serving as the "knowledge base."
    -  **User Query:** A user asks a question via the Gradio interface.
    -  **Retrieval:** The system analyzes the user's query, identifies relevant entities (e.g., employee names, product names), and retrieves corresponding information from the pre-loaded knowledge base.
    -  **Augmented Prompt:** The retrieved information is "stuffed" into the LLM's prompt alongside the original query.
    -  **Generation:** The LLM uses this augmented prompt to generate a highly accurate and contextually relevant response.
- **Example Interaction (using the Gradio Interface):**
    ![Day 18 Gradio RAG Chatbot](https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day18/Implementation.jpg)
---
### **Day 19: Enhancing RAG with LangChain Document Loaders & Text Splitters** 📄✂️
- **Objective:** Deepen the capabilities of the RAG system by integrating **LangChain's document loaders** and advanced **text splitting techniques**. This day focuses on how to efficiently ingest diverse document types, add metadata, and intelligently chunk text for optimal LLM processing. This script is a continuation of Day 18.  
- **Tech Stack:**
    - **LangChain** (`DirectoryLoader`, `TextLoader`, `CharacterTextSplitter`)
- **Highlights:**
    - **Automated Document Ingestion:** Utilizes `DirectoryLoader` to automatically discover and load documents from a specified knowledge base directory (`knowledge-base/`), supporting multiple subfolders (e.g., `company`, `contracts`, `employees`, `products`).
    - **Dynamic Metadata Assignment:** Assigns custom **metadata** (e.g., `doc_type` based on the folder name) to each loaded document, enriching the context available to the LLM.
    - **Intelligent Text Chunking:** Employs `CharacterTextSplitter` to break down large documents into smaller, **overlapping chunks**. This ensures that:
        - Information fits within LLM context windows.
        - Semantic coherence is maintained across chunk boundaries (via `chunk_overlap`).
        - Retrieval of relevant information is more precise.
    - **Chunk Inspection:** Provides insights into the chunking process, showing the number of chunks created, their metadata, and content snippets.
    - **Keyword Search across Chunks:** Demonstrates how to iterate through the created chunks to perform a simple keyword search, highlighting the value of chunking for targeted information retrieval within a RAG pipeline.
- **How it Works:**
    1.  **Load Documents:** The script uses `DirectoryLoader` to load all Markdown (`.md`) files from the `knowledge-base` directory and its subfolders. As documents are loaded, metadata like `source` and `doc_type` (derived from the subfolder name) are automatically added.
    2.  **Split Documents:** Loaded documents are then passed to `CharacterTextSplitter`, which divides them into chunks of a specified `chunk_size` (e.g., 1000 characters) with a defined `chunk_overlap` (e.g., 200 characters). This overlap helps preserve context where information might span across two chunks.
    3.  **Inspect & Search:** The resulting chunks are inspected to verify their structure and content. A basic **keyword search** (e.g., for "CEO") is performed across all chunks to demonstrate how specific information can be isolated after the chunking process.
---
### **Day 20: Vector Embeddings & Visualizing Semantic Space with t-SNE** 📊✨
- **Objective:** This session dives deep into **vector embeddings**, a crucial concept for advanced RAG systems. It covers how text is converted into numerical representations, stored in a **vector database (Chroma)**, and then visualized using **t-SNE** to explore the semantic relationships within our knowledge base. This script is a continuation of Days 18-19. 
- **Tech Stack:**
    - **LangChain** (`OpenAIEmbeddings`, `Chroma`)
    - **OpenAI API** (`text-embedding-ada-002` for embeddings)
    - **ChromaDB** (for persistent vector storage)
    - **Scikit-learn** (`TSNE` for dimensionality reduction)
    - **Plotly** (for interactive 2D and 3D visualizations)
- **Key Concepts Covered:**
    - **Vector Embeddings:** Numerical representations of text that capture semantic meaning. Texts with similar meanings are located closer together in the high-dimensional vector space.
    - **Semantic Search:** Leveraging embeddings to find text chunks that are *conceptually* similar to a query, rather than just keyword matches. This is a core component of RAG.
    - **Chroma Vector Store:** A local, persistent vector database used to store our document chunks and their corresponding embeddings, making them easily retrievable.
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A powerful dimensionality reduction technique used to visualize high-dimensional data, revealing clusters and relationships.
- **Highlights:**
    - **Embedding Generation:** Each processed document chunk from Day 19 is converted into a high-dimensional vector using OpenAI's `text-embedding-ada-002` model.
    - **Vector Database:** The generated embeddings and their associated metadata are stored in a local **ChromaDB** instance.
    - **Dimensionality Inspection:** Confirms that the embeddings have a typical dimensionality (e.g., 1536 dimensions for `text-embedding-ada-002`).
    - **Interactive 2D & 3D Visualizations:**
        - **t-SNE plots** are generated to project the high-dimensional embeddings into an easily interpretable 2D or 3D space.
        - **Color-coding** (by `doc_type`) and **size-scaling** (by chunk length in the 5D visualization) are used to add more dimensions to the plots, helping to identify semantic clusters and relationships between different types of documents.
        - **Hover text** provides detailed information (document type, source path, content preview) for each data point upon interaction.
- **How it Works:**
    1.  **Load & Chunk Documents:** The script re-uses the document loading and chunking logic from Day 19 to prepare the text data.
    2.  **Create Embeddings:** An `OpenAIEmbeddings` model is initialized.
    3.  **Initialize ChromaDB:** A `Chroma` vector store is created (or reset if it already exists) and populated with the document chunks and their embeddings. This process stores the numerical representations on disk.
    4.  **Visualize with t-SNE:**
        - All embeddings are retrieved from Chroma.
        - `TSNE` is applied to reduce the embeddings to 2 or 3 dimensions.
        - `Plotly` is used to create interactive scatter plots, allowing exploration of how different document types (company, contracts, employees, products) cluster together based on their semantic similarity.
- **Visual Models:**
    - **2D Chroma Vector Store Visualization:**
      ![2D Chroma Vector Store Visualization]()
    - **3D Chroma Vector Store Visualization:**
      ![3D Chroma Vector Store Visualization]()
    - **5D Embedding Visualization (3D t-SNE + Color + Size):**
      ![5D Embedding Visualization]()
---
### **Day 21: RAG System - Vector Store Visualization & Chatbot with FAISS** 📊🗣️

- **Objective:** Build a Retrieval-Augmented Generation (RAG) Knowledge Worker using **FAISS (Facebook AI Similarity Search)** for efficient vector storage and retrieval. This script is a continuation of Days 18-20 but uses `FAISS` instead of `Chroma` to handle vector storage & retrieval. `FAISS` is highly optimized for speed, leveraging advanced indexing algorithms & strong GPU acceleration.  
- **Tech Stack:**
    - `LangChain`
    - `OpenAI` ('gpt-4o-mini')
    - `FAISS`
- **Additions from Days 18-20:**
    - **FAISS Vector Store:** Stores embeddings in a `FAISS` index for fast similarity searches.
    - **Conversational RAG:** Implements a LangChain `ConversationalRetrievalChain` with `ChatOpenAI` and `ConversationBufferMemory`. The chatbot retrieves relevant document chunks, augments the LLM's prompt with context, and generates informed responses while maintaining chat history.
    - **Chat Interaction:** The chatbot uses the `FAISS` index to retrieve relevant context for user queries,  augmenting the LLM's response and maintaining conversation history.
---
### **Day 22: RAG System Diagnostics & Chunking Strategy** 🔍

- **Objective:** Diagnose and resolve issues in a Retrieval-Augmented Generation (RAG) system where the Large Language Model (LLM) fails to answer questions due to insufficient context. This script is a continuation of Days 18-21. This focuses on understanding LangChain's internal execution flow using callbacks and optimizing the chunking strategy. 

- **Additions from Days 18-21:** 
    - `LangChain` underneath the hood
    - `StdOutCallbackHandler`

- **Highlights:** 
    - **Diagnosing Retrieval Issues with Callbacks:** In Day 21, the RAG system struggled to answer specific questions like **"Who received the prestigious IIOTY award in 2023?"**. By introducing `callbacks=[StdOutCallbackHandler()]` to the `ConversationalRetrievalChain`, you're able to inspect the exact chunks of information being sent to the LLM. This revealed that the relevant information was not being retrieved, leading to the LLM's inability to answer the question.
    - **Optimizing Chunking Strategy:** The core fix involved adjusting the `k` parameter in the retriever. Previously, the retriever defaulted to a small number of chunks (e.g., `k=4`). By modifying the retriever initialization to `retriever = vectorstore.as_retriever(search_kwargs={"k": 25})`, we significantly increased the number of relevant chunks (to 25) passed to the LLM. This provided the LLM with a much broader context, enabling it to successfully answer the previously unanswerable question about the IIOTY award.
    - **Understanding `k` and its Implications:** Increasing `k` sends more textual data to the LLM, improving accuracy by providing more context. However, this also means higher token usage (potentially increasing API costs), longer processing times, and the need to be mindful of the LLM's context window limits. This day emphasized the importance of tuning retrieval parameters for optimal RAG performance.
---
Day 23: Data Preprocessing for LLM 📊

    Objective: Prepare a large Amazon product dataset ("Appliances") for LLM price prediction through cleaning, feature extraction, and prompt structuring.

    Data:
        Hugging Face datasets (for Amazon-Reviews-2023 Dataset)

    Highlights:
        Prompt Design: Created structured prompts for LLM training and testing.
        Data & Token Visualization: Visualized token and price distributions of the preprocessed data.
---
