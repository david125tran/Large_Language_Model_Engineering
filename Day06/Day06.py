# ------------------------------------ Imports ----------------------------------   
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai
from IPython.display import Markdown, display, update_display
import openai
import google.generativeai as genai

# ------------------------------------ Configure API Keys ----------------------------------
# https://openai.com/api/
# https://console.anthropic.com/ 
# https://ai.google.dev/gemini-api


# ------------------------------------ Load Environment Variables ----------------------------------
# Specify the path to the .env file
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

# Access the API keys stored in the environment variable
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')            # https://openai.com/api/
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')      # https://console.anthropic.com/ 
google_api_key = os.getenv('GOOGLE_API_KEY')            # https://ai.google.dev/gemini-api

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:10]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:10]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:10]}")
else:
    print("Google API Key not set")

print("\n------------------------------------\n")


# ------------------------------------ Connect to OpenAI, Anthropic, and Google Gemini API Platform ----------------------------------   
# Connect to OpenAI
openai = OpenAI()

# Connect to Anthropic
claude = anthropic.Anthropic()

# Connect to Google Gemini
google.generativeai.configure()


# ------------------------------------ OpenAI GPT-3.5-Turbo ----------------------------------
system_message = "You are a comedian that is always joking around."
user_prompt = "Tell me how to make a sandwich."
prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

completion = openai.chat.completions.create(model='gpt-3.5-turbo', messages=prompts)
# Give back one response stored at index 0
print("\n---------------GPT-3.5-Turbo---------------")
print(completion.choices[0].message.content)

# ------------------------------------ OpenAI GPT-4 ----------------------------------
system_message = "You are a comedian that is always joking around."
user_prompt = "Tell me how to make a sandwich."
prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

completion = openai.chat.completions.create(
    model='gpt-4o-mini',
    messages=prompts,
    temperature=0.7 # Temperature setting controls creativity
)   
print("\n---------------GPT-4o-mini---------------")
print(completion.choices[0].message.content)

# ------------------------------------ Anthropic Claude 1 ----------------------------------
# system_message = "You are a comedian that is always joking around."
# user_prompt = "Tell me how to make a sandwich."
# message = claude.messages.create(
#     model="claude-3-7-sonnet-latest",
#     max_tokens=200,
#     temperature=0.7,
#     system=system_message,
#     messages=[
#         {"role": "user", "content": user_prompt},
#     ],
# )
# print("Claude 1:")
# print(message.content[0].text)


# ------------------------------------ Google Gemini ----------------------------------
system_message = "You are a comedian that is always joking around."
user_prompt = "Tell me how to make a sandwich."

gemini = google.generativeai.GenerativeModel(
    model_name='gemini-2.0-flash',
    system_instruction=system_message
)
response = gemini.generate_content(user_prompt)
print("\n---------------Google Gemini---------------")
print(response.text)


# ------------------------------------ Conversation Between ChatGPT and Gemini --------------------------
# Configure OpenAI and Google Gemini API keys
openai.api_key = openai_api_key
genai.configure(api_key=google_api_key)

# System instructions to configure the behavior, tone, and persona of how the systems
# will respond.  
gpt_system = """
You are a five year old child. You are very curious and ask a lot of questions. You 
believe anything you are told. You are so gullible.
"""

gemini_system = """
You are a teacher that is an impulsive liar.  You love telling fake lies to your 
students.  You are very good at it and your students believe everything you say. Make 
up some story involving dragons and unicorns. Add some emojis to the storytelling.
"""

# Create the Gemini chat instance with persistent context
gemini_model_instance = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=gemini_system
)
gemini_chat = gemini_model_instance.start_chat(history=[])

def call_gpt(system_instruction, chat_history):
    """
    Call OpenAI GPT with the given system instruction and pass in conversation history.  
    Every call to GPT requires the full history because each call is stateless or 
    independent. The model does not remember the previous conversation.  The history is 
    passed in as a list of dictionaries.  Each dictionary contains the role (user or
    assistant) and the content of the message.
    """
    # Example message format for ChatGPT passing in it's full conversation history:
    # messages = [
    #     {"role": "system", "content": "You are a friendly assistant that answers questions clearly."},
    #     {"role": "user", "content": "Hi there!"},
    #     {"role": "assistant", "content": "Hello! How can I help you today?"},
    #     {"role": "user", "content": "What's the capital of France?"},
    #     {"role": "assistant", "content": "The capital of France is Paris."},
    #     {"role": "user", "content": "Can you tell me a fun fact about it?"}
    # ]

    # "system" - Sets behavior, tone, or persona of the assistant. 
    # "user" - Represents the input from the human user.  In this case, the human will be Google's 'Gemini' bot.
    # "assistant" - The model's (ChatGPT's) responses to the user.

    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(chat_history)
    completion = openai.chat.completions.create(
        model="gpt-4o", 
        messages=messages
    )

    # Return the assistant's response which is at index 0
    return completion.choices[0].message.content.strip()

def call_gemini(message):
    """
    Call Gemini with a message using persistent chat.  There is no need to pass the full 
    history to Gemini.  It is stored in the chat instance.  The chat instance is created 
    with the system instruction and the history is updated with each message.
    """
    response = gemini_chat.send_message(message)
    return response.text.strip()

# Initialize the chat.
gpt_messages = ["Hi there"]
gemini_messages = ["Hello!"]

print(f"Child (GPT):\n{gpt_messages[0]}\n")
print(f"Teacher (Gemini):\n{gemini_messages[0]}\n")

# Initialize conversation history for GPT since GPT will not have the full history of the 
# conversation.  Gemini will have the full history of the conversation since it is stored in
# the chat instance.  GPT's history is stored in a list of dictionaries.
gpt_history = [
    {"role": "assistant", "content": gpt_messages[0]},
    {"role": "user", "content": gemini_messages[0]}
]

# Main conversation loop
for i in range(5):
    # GPT responds to Gemini.  Feed it the full history + Gemini's most recent message
    gpt_next = call_gpt(gpt_system, gpt_history)
    print(f"Child (GPT):\n{gpt_next}\n")
    # Append the response to the messages and history
    gpt_messages.append(gpt_next)
    # Append the response to the conversation history for ChatGPT as a 'assistant' so that ChatGPT can have the full history
    # of it's responses.  This is important because ChatGPT does not remember the previous conversation.
    gpt_history.append({"role": "assistant", "content": gpt_next})

    # Gemini responds to ChatGPT
    gemini_next = call_gemini(gpt_next)
    print(f"Teacher (Gemini):\n{gemini_next}\n")
    # Append the response to the messages and history
    gemini_messages.append(gemini_next)
    # Append the response to the conversation history for ChatGPT as a 'user' mimicking a human responding to ChatGPT
    gpt_history.append({"role": "user", "content": gemini_next})  # Add to GPT's history for next round

# Save the conversation to HTML
with open(r"C:\Users\Laptop\Desktop\Coding\LLM\Day06\AI Generated Conversation.html", "w", encoding="utf-8") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Generated Conversation</title>
        <style>
            body {
                font-family: Calibri, sans-serif;
            }
        </style>
    </head>
    <body>
    """)
    for i in range(len(gpt_messages)):
        f.write(f"<p><strong style='color:#295F98;'>👶 Child (GPT):</strong> {gpt_messages[i]}</p>\n")
        if i < len(gemini_messages):
            f.write(f"<p><strong style='color:#C96868;'>👨‍🏫 Teacher (Gemini):</strong> {gemini_messages[i]}</p>\n")
        f.write("<br>\n")
    f.write("""
    </body>
    </html>
    """)

print("Conversation saved to HTML file.")