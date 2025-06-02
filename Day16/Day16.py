# ------------------------------------ Packages ----------------------------------
# pip install anthropic
# pip install dotenv
# pip install gradio

# ------------------------------------ Imports ----------------------------------
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic


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
print("\n------------------------------------\n")


# ------------------------------------ Connect to API Platforms ----------------------------------
openai = OpenAI()
claude = anthropic.Anthropic()
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

# Lower cost models:
OPENAI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-3-haiku-20240307"


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
    You are an assistant that reimplements Python code in high-performance {language}.
    Respond only with {language} code; use comments sparingly and do not provide any explanation 
    other than occasional comments. The {language} response needs to produce an identical output 
    in the fastest possible time.
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
    code = code.replace(f"```{language.lower()}", "").replace("```", "")

    # Handle markdown formatting unique to specific languages
    if language.lower() == "c#":
        # Remove the "csharp" tag if it exists
        code = code.replace("csharp", "")
    if language.lower() == "c++":
        # Remove the "cpp" tag if it exists
        code = code.replace("cpp", "")


    # Define the filename based on the target language
    filename = f"C:/Users/Laptop/Desktop/Coding/LLM/Day16/Output/optimized.{language_dict.get(language, 'txt')}"
    with open(filename, "w") as f:
        f.write(code)


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

# Execute the original Python code for baseline output
exec(pi)

# Choose target language for optimization
target_language = prompt_user_for_language()

# Optimize Python code into chosen language
optimize_gpt(pi, target_language)
# optimize_claude(pi, target_language)  

# Run Python again for comparison
exec(pi)



























