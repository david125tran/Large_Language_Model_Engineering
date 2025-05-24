# ------------------------------------ Installations ----------------------------------
# pip install shinywidgets
# pip install shiny


# ------------------------------------ Imports ----------------------------------   
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
import os
from shiny import App, render, ui, reactive


# ------------------------------------ Configure API Key ----------------------------------
# https://openai.com/api/


# ------------------------------------ Constants and Variables ----------------------------------
# https://colorhunt.co/
RED = "#E55050"
BLUE = "#309898"


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


# ------------------------------------ Connect to LLM API Platform(s) ----------------------------------   
openai = OpenAI()
MODEL = 'gpt-4o-mini'


# # ------------------------------------ Chatbot with Gradio Interface ----------------------------------  
# def chat(message, history):
#     """
#     Function to handle chat messages and history.  This function includes context enrichment.
#     Args:
#         message (str): The user message.
#         history (list): The chat history representing the previous turns in the conversation.
#         The history is a list of dictionaries, where each dictionary contains the role (user or assistant) and the content of the message.
#     Returns:
#         generator: A generator that yields the response from the OpenAI API.
#     """
#     # The system message provides instructions to the AI assistant on how to behave and respond.
#     system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
#     the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
#     For example, if the customer says 'I'm looking to buy a hat', \
#     you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
#     Encourage the customer to buy hats if they are unsure what to get."

#     # Context enrichment: Modify the system message based on the user's input
#     relevant_system_message = system_message
#     if 'belt' in message:
#         relevant_system_message += " The store does not sell belts; if you are asked for belts, be sure to point out other items on sale."
    
#     # Constructing the messages list for the API call
#     # The messages list includes the system message, the chat history, and the current user message.
#     messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}]

#     # Calling the OpenAI API (Streaming) to get the assistant's response
#     stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

#     # Initializing an empty string to accumulate the response
#     # The response is built incrementally as the API streams the data.
#     response = ""
#     # This loop iterates over each "chunk" of the streamed response received from the OpenAI API.
#     for chunk in stream:
#         # Each chunk contains a choice, and we extract the content of the delta (the new part of the response).
#         response += chunk.choices[0].delta.content or ''
#         # Instead of returning the full response at the end, yield response returns the current partial response 
#         # each time a new chunk is added. This allows the calling code to display the AI's response progressively, 
#         # as it's being generated, which greatly improves the user experience.
#         yield response

# # Create a Gradio interface for the chat function
# # The Gradio interface allows users to interact with the chat function through a web-based UI.
# gr.ChatInterface(fn=chat, type="messages").launch()


# ------------------------------------ Chatbot with Shiny Interface ----------------------------------
# Define the UI for the Shiny app
app_ui = ui.page_fluid(
    # Add a title to the app
    ui.tags.title("AI Art Idea Generator Assistant"),
    # Custom CSS styles for the chat box and messages
    ui.tags.style("""
        .chat-box {
            border: 1px solid #ccc;
            padding: 10px;
            height: 300px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .user {
            color: #E55050;
            font-weight: bold;
        }
        .assistant {
            color: #309898;  
            font-weight: bold;
        }
        textarea#user_input {
            font-size: 16px;
            padding: 10px;
        }
    """),
    # Page title and header
    ui.h2("Art Idea Generator Assistant"),
    ui.h3("Ask me for ideas for new images."),
    # Chat history output
    ui.output_ui("chat_history"),
    # User input field
    ui.input_text_area("user_input", "Your message:", rows=4, width="100%"),
    # Buttons 
    ui.row(
        ui.input_action_button("send", "Send"),
        ui.input_action_button("clear", "Clear Chat")
    ),
    # Add JavaScript to handle Enter key press for sending messages
    ui.tags.script("""
        document.addEventListener("DOMContentLoaded", function () {
            const input = document.querySelector('textarea[id$="user_input"]');
            const sendBtn = document.querySelector('button[id$="send"]');
            if (input && sendBtn) {
                input.addEventListener("keypress", function (e) {
                    if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();  // Prevent newline and form submission
                        sendBtn.click();
                    }
                });
            }
        });
    """)
)

# Define the server logic for the Shiny app
# The server function handles the logic of the app, including processing user input and generating responses.
def server(input, output, session):
    """Server function to handle chat interactions."""
    # Initialize chat history as a reactive value
    # This will store the chat history as a list of dictionaries, where each dictionary contains the role and content of the message.
    history = reactive.Value([])

    # --------- Chat History Display ---------
    @output
    @render.ui
    def chat_history():
        """Render styled, scrollable chat history."""
        messages = history()
        # Create a div for the chat history
        # Each message is displayed with a different style based on the role (user or assistant).
        chat_display = []
        # Iterate through the chat history and create a div for each message
        # The role is used to determine the class for styling
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            role_class = "user" if role == "user" else "assistant"
            role_label = "User" if role == "user" else "Assistant"
            chat_display.append(ui.div(ui.span(f"{role_label}: ", class_=role_class), content))
        return ui.div(*chat_display, class_="chat-box")

    # --------- Sending Messages and Generating Responses ---------
    @reactive.effect
    @reactive.event(input.send)
    def handle_chat():
        """Handle user input and generate assistant response."""
        user_message = input.user_input().strip()
        # Handle cases where the user message is empty
        # If the user message is empty, do not proceed with generating a response.
        if not user_message:
            return

        system_message = (
            """
            You are an artist helping the user come up with artistic ideas for new images involving their pets.  
            Give them eclectic and creative ideas where you can help suggest certain styles of arts, vivid
            colors, and themes.  Give them ideas for different styles of art, such as nature, abstract, 
            and pop art.  You can also suggest colors, themes, and styles.  The user is looking for ideas for 
            new images involving their pets in which they will have ChatGPT generate the images.

            And so these images can be sci-fi, fantasy, or even surrealism.  Because this ChatGPT mode is 
            capable of drawing anything that the user can imagine.
            """
        )
        # Context enrichment: Modify the system message based on the user's input
        if 'anime' in user_message.lower():
            system_message += " You do not specialize in anime styles, but you can give them ideas for other styles."

        current_history = history()
        # Constructing the messages list for the API call
        messages = [{"role": "system", "content": system_message}] + current_history + [{"role": "user", "content": user_message}]

        response = openai.chat.completions.create(
            # Define the OpenAI model to use
            model="gpt-4",  
            # Define the messages to send to the model
            messages=messages
        )

        # Extract the assistant's reply from the response which is a list of choices at index 0
        # The assistant's reply is the content of the first choice in the response.
        assistant_reply = response.choices[0].message.content
        # Update the chat history with the user message and assistant reply
        history.set(current_history + [{"role": "user", "content": user_message},
                                       {"role": "assistant", "content": assistant_reply}])
        # Clear the user input field after sending the message
        session.send_input_message("user_input", {"value": ""})

    # --------- Clear Chat History ---------
    @reactive.effect
    @reactive.event(input.clear)
    def clear_chat():
        """Clear the chat history."""
        history.set([])


# Create and run the app
app = App(app_ui, server)

# Run the app
if __name__ == "__main__":
    app.run()