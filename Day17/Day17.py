# ------------------------------------ Packages ----------------------------------
# pip install anthropic
# pip install dotenv
# pip install gradio


# ------------------------------------ Imports ----------------------------------
import anthropic
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from openai import OpenAI
import os
from transformers import AutoTokenizer


# ------------------------------------ Configure API Keys / Tokens ----------------------------------
# Specify the path to the .env file
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

# Access the API keys stored in the environment variable
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')            # https://openai.com/api/
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')      # https://console.anthropic.com/ 
google_api_key = os.getenv('GOOGLE_API_KEY')            # https://ai.google.dev/gemini-api
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')      # https://huggingface.co/settings/tokens


print("Checking API Keys...\n")
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

if huggingface_token:
    print(f"Hugging Face Token exists and begins {huggingface_token[:10]}")
else:
    print("Hugging Face Token not set")
print("\n------------------------------------\n")


# ------------------------------------ Connect to API Platforms ----------------------------------
openai = OpenAI()
claude = anthropic.Anthropic()
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

# Lower cost models:
OPENAI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-3-haiku-20240307"

# Code Qwen model
code_qwen = "Qwen/CodeQwen1.5-7B-Chat"

# Login to Hugging Face and configure the Inference Endpoint
# Get the Hugging Face endpoint URLs for the models
CODE_QWEN_URL = "https://qe1ht18838pdue80.us-east-1.aws.endpoints.huggingface.cloud"


# ------------------------------------ Language Dictionary ----------------------------------
language_dict = {
        "AutoIt": "au3",
        "C": "c",
        "C++": "cpp",
        "C#": "cs",
        "Go": "go",
        "Java": "java",
        "JavaScript": "js",
        "Kotlin": "kt",
        "Rust": "rs",
        "PHP": "php",
        "Swift": "swift",
        "Typescript": "ts"
    }


# ----------------------------- Prompt Generators -----------------------------
def prompt_user_for_language():
    print("Choose a target language for optimization:\n")
    languages = list(language_dict.keys())
    for i, lang in enumerate(languages, 1):
        print(f"{i}. {lang}")
    
    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(languages):
                print(f"You chose: {languages[choice - 1]}\n")
                return languages[choice - 1]
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_system_message(language):
    """
    This function generates a system message for the LLM to rewrite Python code in the specified language.
    It includes instructions for the LLM to focus on performance and correctness,
    ensuring that the rewritten code produces identical output in the fastest possible time.
    Args:
        language (str): The target programming language for the optimization.
    Returns:
        str: A formatted system message that includes instructions for the LLM.
    """
    return f"""
    You are a code conversion assistant. Rewrite the following Python code into **only** {language} code.
    - Respond with *only* valid {language} code.
    - Do **not** explain anything.
    - Do **not** include any comments or markdown formatting.
    - Do **not** use phrases like "Here is" or "This code does..."
    """

def user_prompt_for(python_code, language):
    """
    This function generates a user prompt for the LLM to rewrite Python code in the specified language.
    It includes instructions for the LLM to focus on performance and correctness,
    ensuring that the rewritten code produces identical output in the least time.
    Args:
        python_code (str): The Python code to be rewritten in the target language.
        language (str): The target programming language for the optimization.
    Returns:
        str: A formatted user prompt that includes the Python code and instructions for rewriting it in the target language.
    """

    return f"""
    Rewrite this Python code in {language} with the fastest possible implementation that produces 
    identical output in the least time. Respond only with {language} code; do not explain your work 
    other than a few comments. Ensure the output matches exactly and consider performance optimizations.

    {python_code}
    """

def messages_for(python_code, language):
    """
    This function generates a list of messages for the LLM, including a system message and a user prompt.
    Args:
        python_code (str): The Python code to be rewritten in the target language.
        language (str): The target programming language for the optimization.
    Returns:
        list: A list of dictionaries representing the messages for the LLM.
    """

    return [
        {"role": "system", "content": get_system_message(language)},
        {"role": "user", "content": user_prompt_for(python_code, language)}
    ]


# ----------------------------- Output Writer -----------------------------
def write_output(code, language):
    """
    This function writes the generated code to a file with the appropriate extension based on the target language.
    Args:
        code (str): The code to be written to the file.
        language (str): The target programming language for which the code is written.
    Returns:
        None
    """
    language_dict.get(language, "txt")  # fallback to .txt if language unknown

    # Remove Markdown formatting from the code that the LLM might have added
    # code = code.replace(f"```{language.lower()}", "").replace("```", "")
    code = code.split("```")[1]  # Remove any leading markdown code block

    # Handle markdown formatting unique to specific languages
    if language.lower() == "c#":
        # Remove the "csharp" tag if it exists
        code = code.replace("csharp", "")
    elif language.lower() == "c++":
        code = code.replace("cpp", "")
    else:
        code = code.replace(language.lower(), "")


    # Define the filename based on the target language
    filename = f"C:/Users/Laptop/Desktop/Coding/LLM/Day17/Output/optimized.{language_dict.get(language, 'txt')}"
    with open(filename, "w") as f:
        f.write(code)


# ----------------------------- Stream Code Qwen -----------------------------
def stream_code_qwen(python, language):
    """
    This function streams the output of a Code Qwen model hosted on Hugging Face Inference Endpoint.
    It sends a prompt to the model and yields the generated tokens as they arrive.
    Args:
        python (str): The Python code to be optimized.
    Yields:
        str: The generated tokens from the model as they are streamed.
    """

    # Load the tokenizer for the "code_qwen" model (e.g., a code-focused LLM from Hugging Face)
    tokenizer = AutoTokenizer.from_pretrained(code_qwen)
    # Create a list of messages to send to the model
    messages = messages_for(python, "C++")
    # Format the chat messages into a prompt string using the tokenizer's chat template
    # `tokenize=False` means it returns a string, not token IDs
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Create an inference client for the Code Qwen model hosted at CODE_QWEN_URL using an auth token
    client = InferenceClient(CODE_QWEN_URL, token=huggingface_token)
    # Send the prompt to the model for text generation (as a stream, allowing partial outputs)
    stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
    # Stream and yield the generated tokens as they arrive, accumulating them into a result string
    result = ""
    for r in stream:
        result += r.token.text
        yield result    


# ----------------------------- Connect to API Platforms (Optimizers) -----------------------------
def optimize_gpt(python_code, language):
    """
    This function optimizes Python code by sending it to the OpenAI API for rewriting in the specified language.
    It streams the response from the API and prints it to the console in real-time.
    Args:
        python_code (str): The Python code to be optimized.
        language (str): The target programming language for the optimization.
    Returns:
        None
    """
    
    stream = openai.chat.completions.create(
        model=OPENAI_MODEL,  
        messages=messages_for(python_code, language),
        stream=True # Enable streaming to get real-time response and make it look more human-like
    )
    reply = ""
    # Iterate through the streamed response and print each fragment to make it look more human-like
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        print(fragment, end='', flush=True)
    write_output(reply, language)

def optimize_claude(python_code, language):
    """
    This function optimizes Python code by sending it to the Claude API for rewriting in the specified language.
    It streams the response from the API and prints it to the console in real-time.
    Args:
        python_code (str): The Python code to be optimized.
        language (str): The target programming language for the optimization.
    Returns:
        None
    """

    result = claude.messages.stream(
        model=CLAUDE_MODEL,  
        max_tokens=2000,
        system=get_system_message(language),
        messages=[{"role": "user", "content": user_prompt_for(python_code, language)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            print(text, end="", flush=True)
    write_output(reply, language)

def optimize_qwen(python_code, language):
    """
    This function optimizes Python code using a Qwen model hosted on a Hugging Face Inference Endpoint.
    It streams the response and writes the final output to a file.
    
    Args:
        python_code (str): The Python code to be optimized.
        language (str): The target programming language.
    Returns:
        None
    """

    # Prepare input messages using the same structure you use for Claude
    messages = messages_for(python_code, language)

    # Tokenize using the correct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(code_qwen)  # Assuming `code_qwen` is your model ID
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Setup client
    client = InferenceClient(CODE_QWEN_URL, token=huggingface_token)

    # Stream output
    stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
    result = ""
    for r in stream:
        token = r.token.text
        result += token
        print(token, end="", flush=True)

    # Save the result
    write_output(result, language)


# ----------------------------- Example Python Code -----------------------------
pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

# ----------------------------- Execution -----------------------------
# Stream Code Qwen model
# stream = stream_code_qwen(pi, "C++")
# # Print the streamed output from Code Qwen
# for output in stream:
#     print(output, end="", flush=True)

# Execute the original Python code for baseline output
exec(pi)

# Choose target language for optimization
target_language = prompt_user_for_language()

# Optimize Python code into chosen language
# optimize_gpt(pi, target_language)
# optimize_claude(pi, target_language)  
optimize_qwen(pi, target_language)

# Run Python again for comparison
exec(pi)





















