# 🔥 **LLM Engineering Playground: My Journey with Large Language Models** 🔥

### Exploring the Frontier of AI & Crafting Unique Solutions

This repository chronicles my hands-on adventure into Large Language Model (LLM) engineering. While I'm coding along with an online course, I'm deliberately diverging to infuse my own creative twists and explore practical, real-world applications. Expect to see custom integrations, unique AI interactions, and a focus on getting my hands dirty with powerful language models.

---

## 🚀 **My LLM Explorations (By Day)**

Here's a peek into the evolving projects and concepts within this repository:

### **Day 01: Cloud-Powered Conversations & Summaries**
- **Objective:** Harnessing the power of cloud-based LLMs for basic chat and content summarization.
- **Tech Stack:** [OpenAI API Platform](https://platform.openai.com/docs/api-reference/introduction) (using 'gpt-4o-mini').
- **Highlights:** Built a system to automate interactions with language models and extract key information from scraped websites. Imagine instantly summarizing an entire article!

### **Day 02: Taking AI Local with Ollama**
- **Objective:** Enabling local execution of an AI model (LLaMA) for offline capabilities.
- **Tech Stack:** Ollama API, LLaMA model.
- **Highlights:** Successfully set up a local LLM environment, proving that powerful AI isn't always cloud-bound. This opens up possibilities for privacy-focused or resource-constrained applications.

### **Day 05: AI-Powered Portfolio Generation**
- **Objective:** Leveraging AI to dynamically create a markdown portfolio of my GitHub projects.
- **Tech Stack:** BeautifulSoup (for web scraping), OpenAI's ['gpt-4o-mini'](https://platform.openai.com/docs/models/gpt-4o-mini) API.
- **Highlights:** Developed a script that scrapes my GitHub for relevant project links and then uses an LLM to automatically generate a well-formatted markdown portfolio. Automating content creation for showcasing work!

### **Day 06: I configure GPT to be a gullable child and Gemini to be a teacher that loves telling elaborate lies involving Dragons and Unicorns.** 🎭
- **Objective:** Configuring different LLM platforms to adopt contrasting personas and engage in a defined conversation.
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
- **Highlights:** Created an "AI Art Idea Generator Assistant" chatbot specifically designed to help users brainstorm creative ideas for pet-themed AI-generated images. It suggests art styles, colors, and themes!
    - **Example Chat with the OpenAI Chatbot:**
        ![Shiny App Chatbot Interaction](https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day08/AI%20Prompt.jpg)
    - **Putting the Idea into Action (using ChatGPT for Image Generation with the idea):**
        <p align="center">
            <img src="https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day08/ChatGPT%20Prompt.jpg" alt="ChatGPT generated image prompt" width="70%" height="auto">
        </p>
    - **And the Amazing Result!** (Cats with a Salvador Dalí influence!)
        <p align="center">
            <img src="https://github.com/david125tran/Large_Language_Model_Engineering/blob/main/Day08/Cats%20with%20a%20Salvador%20Dali%20Influence.jpg" alt="AI-generated cats with surreal influence" width="50%" height="auto">
        </p>

### **Day 09: Advanced LLM Tooling** 🛠️
- **Objective:** Developing an AI Chatbot as a Gradio App, focusing on maintaining conversational context and equipping the bot with a custom tool to add additional knowledge to the bot.  This bot provides nutritional data for fruits from a specific source (through API integration) and alerts the end user if it is unable to find data through the API call.  Integrating the chat bot with API interactions is a very powerful tool because the data source can be controlled.  
- **Tech Stack:** OpenAI API, Gradio App, API integration (Fruitvice API)
- **Highlights:** Robust API integration gracefully handling when a fruit is not found in the API call.

  
---
