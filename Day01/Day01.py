# ---------------------------------- Terminal Installations ----------------------------------
# py -m pip install python-dotenv
# py -m pip install requests
# py -m pip install beautifulsoup4
# py -m pip install openai
# py -m pip install openai[embeddings]
# py -m pip install openai[chat]
# py -m pip install openai[whisper]
# py -m pip install openai[all]
# py -m pip install openai[all] --upgrade
# py -m pip install openai[all] --upgrade --force-reinstall


# ------------------------------------ Imports ----------------------------------   
import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI


# ------------------------------------ Functions ---------------------------------- 
# A class to represent a Webpage
class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \n\n"
    user_prompt += website.text
    return user_prompt


def summarize(url, messages):
    website = Website(url)
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages
    )
    return response.choices[0].message.content

# ------------------------------------ Connect to OpenAPI API Platform ----------------------------------  
# OpenAi API Platform Username: david112tran@gmail.com 
# https://platform.openai.com/

# Specify the path to your .env file
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Day01\.env"

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

# print(api_key)

# Create an OpenAI instance
openai = OpenAI(api_key=api_key)


# ------------------------------------ Messaging OpenAi's Frontier Model ----------------------------------   
# Create a chat 
message = "What is the current weather in Raleigh, NC?"

# Send the message to the Frontier chat model (gpt-4o-mini)
# response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content":message}])
# print(response.choices[0].message.content)


# ------------------------------------ Web Scraping and Summarizing the Page w/Open Ai's Frontier Model ----------------------------------
# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

# Example URL
david_tran_github = Website("https://github.com/david125tran")
# print(david_tran_github.title)  
# print(david_tran_github.text)

message = f"Summarize the following text:\n{david_tran_github.text}"
response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content":message}])
print(response.choices[0].message.content)

# Return Message:

# The text provides an overview of David Tran, a software 
# developer based in Durham, NC, highlighting his skills, 
# projects, and technologies used. He specializes in building 
# automated solutions and has experience in scientific computing, 
# database management, and system automation. His GitHub profile 
# lists various projects he has worked on, such as a CRUD web 
# app, data conversion tools, REST APIs using Flask and Django, 
# and web scraping utilities, among others. David is skilled in 
# multiple programming languages, including Python, SQL, JavaScript, 
# and C#, and is passionate about backend development and automation.


# ------------------------------------ Chatting with the Model (Example 1: Mom Helping w/Homework) ----------------------------------
# Models like GPT4o have been trained to receive instructions in a particular way.
# They expect to receive:
#   *A system prompt that tells them what task they are performing and what tone they should use
#   *A user prompt -- the conversation starter that they should reply to

# The system prompt is a special message that tells the model what to do.

# Define our system prompt - you can experiment with this later, changing the last sentence to 'Respond in markdown in Spanish."

print("Example 1: Mom helping with homework:\n")
system_prompt = """
                You are an assistant that helps explain stuff to little 
                children.  Talk as if you are explaining to a 5 year old.  Use simple words 
                and short sentences.  Use emojis to help explain things.
                """

# The API from OpenAI expects to receive messages in a particular structure. Many of the other APIs share this structure:
# A list of two dictionaries, each with a role and content:
[
    {"role": "system", "content": "system message goes here"},
    {"role": "user", "content": "user message goes here"}
]

# The system message is a special message that tells the model what to do.  They are to act like a nice mom who helps their
# kids with their homework.  The user message is the question that the model is to answer.
messages = [
    {"role": "system", "content": "You are a nice mom who helps kids with their homework."},
    {"role": "user", "content": "What is 2 + 2?"}
]

# Calling OpenAI with system and user messages:
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)
# Response Message:
# 2 + 2 equals 4! Great job on your math!

# ------------------------------------ Chatting with the Model (Example 2: Sarcastic Assistant) ----------------------------------
print("\nExample 2: Sarcastic Assistant:\n")

# Example URL
david_tran_github = Website("https://github.com/david125tran")

system_prompt = """
                You are an extremely sarcastic assistant.  Use sarcasm to answer the user's question.
                """


messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt_for(david_tran_github)}
]

print(summarize("https://github.com/david125tran", messages))

# Response Message:
# ```markdown
# # Summary of David Tran's GitHub Profile

# Welcome to the riveting world of **david125tran**—because, obviously, you have nothing better to do than browse through someone's repositories and projects. 

# - **About David**: This is a software developer (who apparently thinks building efficient, automated solutions is a big deal) from Durham, NC. He has a wild collection of skills, like Python, SQL, and even HTML/CSS. Groundbreaking, right?

# - **Repositories**: A whopping **31 repositories** to choose from! It’s like a candy store... if all the candy was slightly stale and just okay. Some highlights include:
#   - **ASP.NET Core Web App**: Because nothing screams originality like a CRUD app.
#   - **Book to Audio Converter**: Who needs to read anyway?
#   - **Django REST API**: Because every dev needs a REST API in their life.

# - **Achievements**: It looks like he has four followers. Wow, such popularity in the vast ocean of GitHub!

# - **Contact Information**: Just in case visiting his GitHub wasn’t intrusive enough, you can also email him. What a generous offer!

# Happy browsing, because I know you’re just thrilled to dive into David's world of databases and APIs! 🥱
# ```