# ---------------------------------- Terminal Installations ----------------------------------
# py -m pip install requests
# py -m pip install beautifulsoup4
# py -m pip install ipykernel
# py -m ipykernel install --user --name=ollama
# py -m pip install pyttsx3


# ------------------------------------ Imports ----------------------------------   
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
import pyttsx3
import requests


# ------------------------------------ Constants ----------------------------------
# Ollama is running here: http://localhost:11434
# Ollama API endpoint and headers
# Note: Make sure the Ollama API is running on your local machine.
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"


# ------------------------------------ Functions ---------------------------------- 
def message_ollama(message):
    """
    Send a message to the Ollama API and return the response.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": message}],
        "stream": False
    }

    response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
    return response.json()['message']['content']

def speak(message):
    """
    Convert text to speech using pyttsx3.
    """
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()


# ------------------------------------ Main Loop ---------------------------------- 
message = "Hello, Ollama!, what is the current weather in Durham, NC?"

response = message_ollama(message)

print(response)
speak(response)