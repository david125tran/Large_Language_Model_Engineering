# ------------------------------------ Imports ----------------------------------   
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
import gradio as gr
import requests


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


# ------------------------------------ Function to Fetch Fruit Nutrition Data through API Call ----------------------------------
def get_fruit_nutrition(fruit_name: str) -> dict | None:
    """
    Fetches nutritional data for a given fruit name from the Fruityvice API.

    Args:
        fruit_name (str): The name of the fruit to search for (e.g., "apple", "banana").

    Returns:
        dict | None: A dictionary containing the fruit's nutritional data if found,
                     otherwise None.
    """
    # URL for the Fruityvice API
    URL = "https://fruityvice.com/api/fruit/"

    # Normalize the fruit name
    fruit_name = fruit_name.strip().lower()  

    # Remove the trailing 's' if present so that the API can find the fruit
    if (fruit_name[-1] == 's'):
        fruit_name = fruit_name[:-1]  

    # Construct the full URL for the specific fruit.  Ex. "https://fruityvice.com/api/fruit/apple"
    url = f"{URL}{fruit_name}"

    print(f"Attempting to fetch data for: {fruit_name} from {url}")

    try:
        # Make the GET request to the Fruityvice API
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            fruit_data = response.json()
            return fruit_data
        elif response.status_code == 404:
            print(f"Error: Fruit '{fruit_name}' not found. Status Code: {response.status_code}")
            return None
        else:
            print(f"Error fetching data for '{fruit_name}'. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    

# ------------------------------------ OpenAI - Function Schema ----------------------------------
# This is a function schema that describes a tool the chatbot can use.  In this case, I am training
# the chatbot how to use my get_fruit_nutrition() function.  The function schema is a dictionary 
# that describes the function and its parameters.  The function schema is passed to the OpenAI API 
# when creating the chatbot.  The chatbot will use this schema to understand how to call the function.

fruit_function = {
    "name": "get_fruit_nutrition",
    "description": "Get the nutritional data for a specific fruit. Call this you are asked information about a fruit.",
    "parameters": {
        "type": "object",
        "properties": {
            "fruit_name": {
                "type": "string",
                # Train the model to pass a single fruit name.  The API call will fail if the fruit name is plural.
                "description": "The name of a single fruit (e.g., 'apple', not 'apples')." 
            },
        },
        "required": ["fruit_name"],
        "additionalProperties": False
    }
}

# Register the fruit_function as a tool that the chatbot can use.  The tools list is passed to 
# the OpenAI API when creating the chatbot.
tools = [{"type": "function", "function": fruit_function}]

def chat(message, history):
    system_message = """
    You are a helpful assistant for that knows about fruits and their nutritional data.
    Give short, courteous answers, no more than 3 sentences on the nutritional data of fruits.
    If you don't know the answer or the fruit is not found in the database,
    respond by saying that you could not find nutritional information for that fruit from 
    'Fruityvice' but tell them that you do have information on it from another source and tell
    them what you do know.
    """

    # Add the system message to the history
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    # Call the OpenAI API to get the response from the chatbot and equip it with the tools (To find fruit nutrition data)
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    # Example response format from the OpenAI API
    # ChatCompletion(id='chatcmpl-BakQf4IPuuIyBgBgeNS7Pch09cLbJ', choices=[Choice(finish_reason='tool_calls', index=0, 
    # logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=[], audio=None, 
    # function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_nn4aGZokyyr5dEKb1DFXAFvv', 
    # function=Function(arguments='{"fruit_name":"kiwi"}', name='get_fruit_nutrition'), type='function')]))], 
    # created=1748097981, model='gpt-4o-mini-2024-07-18', object='chat.completion', service_tier='default', 
    # system_fingerprint='fp_62a23a81ef', usage=CompletionUsage(completion_tokens=20, prompt_tokens=186, total_tokens=206, 
    # completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, 
    # rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))

    # Check if the response contains a tool call (when the user asks about a fruit).  If it does, handle the tool call and get the response.
    # When GPT-4o-mini doesn't have an answer yet, it will stop and call one of the tools and provides an output.  
    if response.choices[0].finish_reason=="tool_calls":
        print("Tool call detected")
        # Get the tool call from the response
        message = response.choices[0].message
        # Call the tool (get_fruit_nutrition()) and get the response
        response = handle_tool_call(message)
        # Append the message to the chat history
        messages.append(message)
        # Append the tool call response to the chat history (The response from the get_fruit_nutrition() function)
        messages.append(response)

        # Example of the messages list after the tool call (We pass to OpenAI the tool call response (fruit data from get_fruit_nutrition())):

        # [{'role': 'system', 'content': "\n    You are a helpful assistant for that knows about fruits and their nutritional data.\n    
        # Give short, courteous answers, no more than 3 sentences on the nutritional data of fruits.\n    If you don't know the answer 
        # or the fruit is not found in the database,\n    respond by saying that you could not find nutritional information for that 
        # fruit from \n    'Fruityvice' but tell them that you do have information on it from another source and tell\n    them what 
        # you do know.\n    "}, {'role': 'user', 'content': 'What do you know about kiwi?'}, ChatCompletionMessage(content=None, 
        # refusal=None, role='assistant', annotations=[], audio=None, function_call=None, 
        # tool_calls=[ChatCompletionMessageToolCall(id='call_2Pbv55god9lnIUFB0fNtdZrf', function=Function(arguments='{"fruit_name":"kiwi"}', 
        # name='get_fruit_nutrition'), type='function')]), {'role': 'tool', 'content': '{"name": "Kiwi", "id": 66, "family": "Actinidiaceae", 
        # "order": "Struthioniformes", "genus": "Apteryx", "nutritions": {"calories": 61, "fat": 0.5, "sugar": 9.0, "carbohydrates": 15.0, "protein": 1.1}}', 
        # 'tool_call_id': 'call_2Pbv55god9lnIUFB0fNtdZrf'}]

        # Call the OpenAI API again to get the final response from the chatbot passing in the messages list containing
        # the tool call response (fruit data from get_fruit_nutrition()).
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    # Get the response from the chatbot which is at the 0th index of the choices list
    # The response is a dictionary with the role and content of the message
    return response.choices[0].message.content


def handle_tool_call(message):
    """
    This function handles the tool call from the chatbot and gets the response from the get_fruit_nutrition() function.
    It takes the message as input, extracts the fruit name from the tool call, and calls the function to get the nutritional data.
    It returns the response from the function call as a dictionary with the role and content of the message.
    """
    # Extract the tool call from the message
    tool_call = message.tool_calls[0]
    # Get the fruit name from the tool call.  Ex: {'fruit_name': 'kiwi'}
    arguments = json.loads(tool_call.function.arguments)
    # Get the fruit name from the arguments.  Ex: 'kiwi'
    fruit_name = arguments.get('fruit_name')
    # Call the get_fruit_nutrition() function to get the nutritional data for the fruit
    fruit_data = get_fruit_nutrition(fruit_name)

    # Check if the fruit data is None or contains an error from the API call (not in the database)
    if (fruit_data is None) or ("Error" in fruit_data):
        response = {
            "role": "tool",
            "content": json.dumps({"error": f"Fruit '{fruit_name}' not found."}),
            "tool_call_id": tool_call.id
        }
        return response

    # If the fruit data is found, format it as a string
    response = {
        "role": "tool",
        "content": json.dumps(fruit_data),
        "tool_call_id": tool_call.id
    }
    return response

# ------------------------------------ Gradio Interface ----------------------------------
with gr.Blocks(title="Fruit Nutritional Data AI Chatbot") as webserver:
    gr.Markdown(
        """
        # 🥝 Fruit Nutritional Data AI Bot  
        _Ask me about any fruit's nutrition facts!_  
        I use real-time data from the **Fruityvice API** and OpenAI to help you learn about fruits.
        """
    )
    gr.Markdown(
        """
        ## Instructions
        1. Type your question in the chat box.
        2. Press Enter or click the Send button to submit your question.
        3. The bot will respond with nutritional data for the fruit you asked about.
        4. If the fruit is not found in the FruityVice database, the bot will let you know and provide information from another source.
        """
    )
    # Create the chat interface
    gr.ChatInterface(fn=chat, type="messages")

webserver.launch()