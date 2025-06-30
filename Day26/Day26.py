# ------------------------------------ LLM Engineering Project: Human-in-the-Loop Evaluation ----------------------------------
"""
This script supports a human-in-the-loop evaluation workflow for price estimation of ecommerce products.

The purpose is to export LLM-generated prompts to a CSV for human price labeling, collect human responses,
and compare human predictions to ground truth using a custom testing framework.

💡 Key Workflow Steps:

1. **Prompt Export for Human Labeling**
    - Export 250 LLM-formatted prompts and product descriptions to a CSV (`human_input.csv`)
    - Humans review the products and input price estimates into a second CSV (`human_output.csv`)

2. **Human Prediction Parsing**
    - Read human-labeled prices, with validation and cleaning
    - Align predictions with the test set for downstream comparison

3. **Human vs Model Evaluation**
    - Define a `human_pricer()` function for injecting human predictions
    - Run the `Tester` class to evaluate human error vs true prices (RMSLE)

4. **LLM Prompt Engineering and Model Comparison**
    - Prepare clean and instructive LLM prompts for price prediction
    - Call various LLMs (e.g., GPT-4o, Claude) and compare their performance
    - Normalize outputs with a custom `get_price()` parser for consistency

🔬 Use Case:
This script is ideal for benchmarking human vs LLM price prediction accuracy
and refining prompt quality before fine-tuning models on structured data.

Human-in-the-loop (HITL) evaluation is critical in machine learning and LLM 
development because it brings human judgment, oversight, and refinement into 
systems that would otherwise rely solely on automation. 

This is not about evaluating the model against human-labeled ground truth — 
it's about evaluating the human (and the LLM) against the actual price (item.price).

"""


# ------------------------------------ Package Install Instructions ----------------------------------
# pip install anthropic chromadb dotenv matplotlib pandas scikit-learn gensim transformers


# ------------------------------------ Imports ----------------------------------
import csv
import os
import re
import math
import json
import random
from dotenv import load_dotenv
from huggingface_hub import login
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import Counter
from openai import OpenAI
from anthropic import Anthropic


# ------------------------------------ Log In to LLM API Platforms ----------------------------------
# Load API Keys from local .env file (for LLM fine-tuning later)
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"
load_dotenv(dotenv_path=env_path, override=True)

# Print out available API keys for safety check
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
# google_api_key = os.getenv('GOOGLE_API_KEY')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

print("Checking API Keys...\n")
if openai_api_key: print(f"OpenAI Key found: {openai_api_key[:10]}...")
if anthropic_api_key: print(f"Anthropic Key found: {anthropic_api_key[:10]}...")
# if google_api_key: print(f"Google Key found: {google_api_key[:10]}...")
if huggingface_token: print(f"HuggingFace Token found: {huggingface_token[:10]}...")

print("\n------------------------------------\n")
# Log into Hugging Face (necessary for items.py tokenizer)
login(huggingface_token, add_to_git_credential=True)
# Import Item and Tester class AFTER huggingface login (due to tokenizer auth)
from items import Item
from testing import Tester

openai = OpenAI()
claude = Anthropic()


# ------------------------------------ Extract *.pkl Files ----------------------------------
# Load previously pre-processed datasets (pickle files)
with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day26\train.pkl', 'rb') as file:
    train = pickle.load(file)
with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day26\test.pkl', 'rb') as file:
    test = pickle.load(file)


# ------------------------------------ Data Exploration ----------------------------------
# This section demonstrates basic data export and human-in-the-loop feedback for evaluation.
# The script exports test prompts for manual price labeling, then reads human responses back in.

# Write the test set to a CSV
with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day26\human_input.csv', 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Header Row
    writer.writerow(["Prompt", "Description", "Price"])  # optional header row
    # Write each test item as a row for human input 250x
    for t in test[:250]:
        # print(vars(t))
        prompt = t.test_prompt().replace("\n", " ")
        title = getattr(t, 'title', 'N/A')  # fallback if `description` is missing
        writer.writerow([prompt, title, "?"])

# Notify user to fill in the CSV file
while True:
    user_input = input("Fill in your predicted prices for the 250 items.  Type 'Y' when you're ready to continue: ").strip().upper()
    if user_input == 'Y':
        break

# Read the human predictions back in
human_predictions = []
with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day26\human_output.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if len(row) < 2 or row[1].strip() == '':
            print(f"⚠️ Skipping row {i+1}: invalid or empty")
            continue
        try:
            human_predictions.append(float(row[1]))
        except ValueError:
            print(f"❌ Row {i+1} has non-numeric data: {row[1]}")

def human_pricer(item):
    """
    Return the human price prediction for a given item.
    """
    idx = test.index(item)
    return human_predictions[idx]

# Run the evaluation using the human pricer and the test set
Tester.test(human_pricer, test)



# ------------------------------------ LLM Prompt Engineering ----------------------------------
# First let's work on a good prompt for a Frontier model
# Notice that I'm removing the " to the nearest dollar"
# When we train our own models, we'll need to make the problem as easy as possible, 
# but a Frontier model needs no such simplification.

def messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, don't give an explanation"
    # Replace the " to the nearest dollar" part to make it more general
    # This is important for training our own models, as we want them to learn the full
    # range of price estimation, not just rounding to the nearest dollar.
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        # The assistant's response is expected to be just the price, without any explanation.
        # We can say, 'Price is $' to indicate the expected format so that it is likely to 
        # give us the next token as the price.
        {"role": "assistant", "content": "Price is $"}
    ]

print(f"Example message:\n{messages_for(test[0])}")
# Example message:
# [{'role': 'system', 'content': "You estimate prices of items. Reply only with the price, don't give an explanation"}, 
#  {'role': 'user', 'content': "How much does this cost?\n\nOEM AC Compressor w/A/C Repair Kit For Ford F150 F-150 V8 & Lincoln Mark LT 2007 2008 - BuyAutoParts NEW\nAs one of the world's largest automotive parts suppliers, our parts are trusted every day by mechanics and vehicle owners worldwide. This A/C Compressor and Components Kit is manufactured and tested to the strictest OE standards for unparalleled performance. Built for trouble-free ownership and 100% visually inspected and quality tested, this A/C Compressor and Components Kit is backed by our 100% satisfaction guarantee. Guaranteed Exact Fit for easy installation 100% BRAND NEW, premium ISO/TS 16949 quality - tested to meet or exceed OEM specifications Engineered for superior durability, backed by industry-leading unlimited-mileage warranty Included in this K"}, 
#  {'role': 'assistant', 'content': 'Price is $'}]


def get_price(s):
    """
    Extract a price from a string.  To handle cases where the LLM response with more text than just the price. 
    We use this to rip out the price from the LLM response.
    """
    s = s.replace('$','').replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0

print(get_price("The price is roughly $99.99 because blah blah"))
# Output: 99.99


# ------------------------------------ LLM Model Calls (gpt_4o_mini) ----------------------------------
def gpt_4o_mini(item):
    """
    Call the GPT-4o-mini model to estimate the price of an item.
    This function constructs a message for the model and returns the estimated price.
    The model is expected to respond with a price in the format "Price is $<amount>".
    If the model responds with more text, we extract the price using the get_price function
    """

    # Create the messages for the model
    # The messages include a system message, a user prompt, and an assistant response template.
    # The user prompt is the test prompt for the item, and the assistant response is expected
    # to be just the price in the format "Price is $<amount>".
    # The seed is set to 42 for reproducibility, and max_tokens is set to 5 to limit the response length.
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages_for(item),
        seed=42,
        max_tokens=5
    )
    reply = response.choices[0].message.content
    # Clean up the reply to ensure it only contains the price
    price = get_price(reply)
    return price

print(f"\nThe price is: ${test[0].price}")                  # The actual price is: $374.41
print(f"gpt_40_mini prediction: ${gpt_4o_mini(test[0])}")   # gpt_40_mini prediction: $210.0


# ------------------------------------ LLM Frontier Model (gpt_4o_frontier) ----------------------------------
def gpt_4o_frontier(item):
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06", 
        messages=messages_for(item),
        seed=42,
        max_tokens=5
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# Predict the price using the GPT-4o Frontier model
Tester.test(gpt_4o_frontier, test)


# ------------------------------------ LLM Claude 3.5 Sonnet Model (claude_3_point_5_sonnet) ----------------------------------
# def claude_3_point_5_sonnet(item):
#     messages = messages_for(item)
#     system_message = messages[0]['content']
#     messages = messages[1:]
#     response = claude.messages.create(
#         model="claude-3-5-sonnet-20240620",
#         max_tokens=5,
#         system=system_message,
#         messages=messages
#     )
#     reply = response.content[0].text
#     return get_price(reply)

# # Predict the price using the Claude 3.5 Sonnet model
# Tester.test(claude_3_point_5_sonnet, test)

