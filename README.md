# 🔥 **LLM Engineering Playground** 🔥

### Exploring the Frontier of AI & Crafting Unique Solutions

This repository chronicles my hands-on adventure into Large Language Model (LLM) engineering. While I'm coding along with an online course, I'm deliberately diverging to infuse my own creative twists and explore practical, real-world applications. Expect to see custom integrations, unique AI interactions, and a focus on getting my hands dirty with powerful language models.

---

## 🚀 **My LLM Explorations (By Day)**

Here's a peek into the evolving projects and concepts within this repository:

### **Day 01: Cloud-Powered Conversations & Summaries** 🗪 
- **Objective:** Harnessing the power of cloud-based LLMs for basic chat and content summarization.
- **Tech Stack:** [OpenAI API Platform](https://platform.openai.com/docs/api-reference/introduction) (using 'gpt-4o-mini').
- **Highlights:** Built a system to automate interactions with language models and extract key information from scraped websites. Imagine instantly summarizing an entire article!

### **Day 02: Configuring AI Chatbot Locally with Ollama** 🦙 
- **Objective:** Enabling local execution of an AI model (LLaMA) for offline capabilities.
- **Tech Stack:** Ollama API, LLaMA model.
- **Highlights:** Successfully set up a local LLM environment, proving that powerful AI isn't always cloud-bound. This opens up possibilities for privacy-focused or resource-constrained applications.

### **Day 05: AI-Powered Portfolio Generation** 🌐 
- **Objective:** Leveraging AI to dynamically create a markdown portfolio of my GitHub projects.
- **Tech Stack:** BeautifulSoup (for web scraping), OpenAI's ['gpt-4o-mini'](https://platform.openai.com/docs/models/gpt-4o-mini) API.
- **Highlights:** Developed a script that scrapes my GitHub for relevant project links and then uses an LLM to automatically generate a well-formatted markdown portfolio. Automating content creation for showcasing work!

### **Day 06: Configuring different LLM platforms to engage in a defined conversation together** 🎭
- **Objective:** Configuring different LLM platforms to adopt contrasting personas and engage in a defined conversation.  I configure GPT to be a gullable child and Gemini to be a teacher that loves telling elaborate lies involving Dragons and Unicorns.  I make these AI platforms talk to one another.  
- **Tech Stack:** OpenAI's 'GPT-4o', Google's Gemini 1.5 Flash.
- **Highlights:** Witness a hilarious dialogue between a "gullible child" GPT and a "teacher who loves telling elaborate lies involving Dragons and Unicorns" Gemini. Overcame GPT's statelessness by maintaining conversation history. The entire dialog is saved to an HTML file for easy viewing!
    - 🔗 [View the conversation HTML here!](https://html.onlineviewer.net/) (Paste the generated HTML content here for easier viewing)

### **Day 07: Gradio Web App - AI Photo Editor** ✨
- **Objective:** Creating a simple web application for AI-powered photo editing.
- **Tech Stack:** Gradio, OpenAI's 'dall-e-2' model.
- **Highlights:** Built an interactive web app where users can input text prompts to dynamically edit images using DALL-E 2.
    - ![Gradio Web Server in Action](https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day07/Gradio%20Web%20Server.png)

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

### **Day 09: Advanced LLM Tooling** 🛠️
- **Objective:** Developing an AI Chatbot as a Gradio App, focusing on maintaining conversational context and equipping the bot with a custom tool to add additional knowledge to the bot.  This bot provides nutritional data for fruits from a specific source (through API integration) and alerts the end user if it is unable to find data through the API call.  I used a free fruit data API that doesn't require authentication just to get the code up & running faster.  But for proof of concept, integrating the chat bot with API interactions is a very powerful tool because the data source can be controlled or the LLM can be trained with additional data.  
- **Tech Stack:** OpenAI API, Gradio App, API integration (Fruitvice API)
- **Highlights:** Robust API integration gracefully handling when a fruit is not found in the API call.
- **Example Chat with the OpenAI Chatbot:**
        ![Gradio App Chatbot Interaction](https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day09/Gradio%20Fruit%20Chatbot.jpg)
  
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

 
### **Day 13: Tokenizers - Exploring LLM Inputs** 🧠 
- **Objective:** Understand how LLMs process text by encoding and decoding with various tokenizers.
- **Tech Stack:** 
    - Hugging Face Transformers, Colab, Meta’s LLaMA 3.1, Phi-3, Qwen2, Starcoder2.
- **Highlights:**  
    - Used Hugging Face open-source models to tokenize text and code, applied chat templates, and compared token outputs across multiple models.


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
     

### **Day 15: Audio to Text Summarization** 🎧
- **Objective:** This notebook transforms audio into structured meeting minutes using advanced AI.  
- **Tech Stack:** 
    - 🤫 **Whisper-1** by OpenAI (For speech-to-text transcription)
    - 🦙 **LLaMA 3.1** by Meta (For summarizing text & extract key details)
    - BitsAndBytes (for 4-bit quantization)
- **Highlights:**
    - Audio transcription, meeting minutes generation, quantization for efficiency, streamed output (demonstrate how to stream LLM's output for a more interactive human-like experience), & Google Drive integration.  

### **Day 16: Code Alchemist — Python to Lightning-Fast Languages** ⚡
- **Objective:** Transform Python scripts into high-performance equivalents in languages like C++, Rust, Go, and more using state-of-the-art LLMs.
- **Tech Stack:** 
  - [OpenAI API](https://platform.openai.com/docs/api-reference/introduction)
  - [Anthropic Claude API](https://console.anthropic.com/)
- **Highlights:**  
  - Supports multiple LLMs (OpenAI + Claude) for redundancy and experimentation  
  - Converts Python into various target languages with a focus on **performance** and **output fidelity**  
  - Cleans up LLM responses (e.g. removes markdown artifacts, formats extensions)  

---
