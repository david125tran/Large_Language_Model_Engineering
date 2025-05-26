# ------------------------------------ Imports ----------------------------------   
import base64
from io import BytesIO
import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import webbrowser
import uuid


# ------------------------------------ Configure API Key ----------------------------------
# https://openai.com/api/


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


# ------------------------------------ Connect to LLM API Platform ----------------------------------   
openai = OpenAI()
MODEL = 'gpt-4o-mini'


# ------------------------------------ Image Generation ----------------------------------
# System prompt
system_message = """
You are a helpful AI artist. When a user describes a scene or asks for an image, respond with a short confirmation or description.
The system will automatically generate the image based on the user's latest message, so you do NOT need to say you cannot generate images.
Just assume the image will be created, and describe it if needed.
"""

def artist(prompt):
    """
    Generate an image based on the provided prompt using OpenAI's DALL-E model.
    Saves the image as a JPEG with a unique ID in the 'Downloads' folder next to this script.
    
    Args:
        prompt (str): The text prompt to generate the image.
    Returns:
        Image: The generated image.
    """
    # Generate image from OpenAI
    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    # Convert the base64 string to bytes
    image_base64 = image_response.data[0].b64_json
    # Decode the base64 string
    image_data = base64.b64decode(image_base64)
    # Create a PIL image from the bytes
    image = Image.open(BytesIO(image_data))

    # Create Downloads directory relative to script if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(script_dir, "Downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # Generate unique filename
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(downloads_dir, filename)

    # Save image as JPEG
    image.convert("RGB").save(save_path, format="JPEG")
    print(f"Image saved to: {save_path}")

    return image

# ------------------------------------ Text-to-Speech ----------------------------------
def talker(message):
    """
    Create speech from text using OpenAI's TTS model and play it.
    """
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message
    )

    # Save audio to temp file and open in default player
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    # Open in system player (Windows, Mac, Linux)
    webbrowser.open(f"file://{tmp_file_path}")


# ------------------------------------ Chat ----------------------------------
# Chat function to process user input and return response + image
def chat(history):
    """
    Process the chat history and generate a response using OpenAI's GPT-4 model.
    Args:   
        history (list): List of chat messages.
    Returns:
        tuple: Updated chat history and generated image.
    """

    # Add the system message to the history
    messages = [{"role": "system", "content": system_message}] + history
    # Call the OpenAI API to get the response from the chatbot
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    # Extract the assistant's reply and add it to the history
    reply = response.choices[0].message.content
    # Append the assistant's reply to the history
    history.append({"role": "assistant", "content": reply})

    # Initialize image to None
    image = None
    # Find the last user message in the history.  We reverse the history to find the last user message
    last_user_message = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), None)
    # If a last user message is found, generate the image
    if last_user_message:
        try:
            # Generate image from the last user message
            image = artist(last_user_message)
        except Exception as e:
            print(f"Image generation failed: {e}")

    # Try to speak reply
    try:
        talker(reply)
    except Exception as e:
        print(f"TTS failed: {e}")

    return history, image


# ------------------------------------ Gradio UI ----------------------------------
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history):
        """
        This function processes the user input and appends it to the chat history. It handles 
        the case where the message is too long for the "dall-e-3" model and provides a warning.
        """
        if len(message) > 400:
            warning = f"⚠️ Your message is too long ({len(message)} characters). Please limit it to 400 characters."
            # Append the warning to the history
            history.append({"role": "assistant", "content": warning})
            return "", history
        
        # Append the user message to the history
        history.append({"role": "user", "content": message})
        return "", history

    # Set up the event handlers
    # When the user submits a message, process it and update the chatbot
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]) \
        .then(chat, inputs=chatbot, outputs=[chatbot, image_output])
    # When the user clicks the clear button, clear the chatbot history
    clear.click(lambda: [], outputs=chatbot, queue=False)

# ------------------------------------ Launch the UI ----------------------------------
ui.launch(inbrowser=True)