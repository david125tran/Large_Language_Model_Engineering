# ------------------------------------ Fine-Tuning LLMs (OpenAI): Preparing Data, Training, & Evaluation ----------------------------------
"""
This script demonstrates a complete pipeline for fine-tuning an OpenAI GPT model (e.g., `gpt-4o-mini-2024-07-18`) to estimate product prices
from text descriptions. It uses OpenAI's supervised fine-tuning interface and includes integration with Weights & Biases (wandb) for
experiment tracking.

📌 **Project Goal**
- Build a custom price estimator LLM by fine-tuning a small OpenAI model on structured `Item` data containing product descriptions and prices.

🚀 **Workflow**
1. **Data Preparation**
   - Load product description data from pickled Python objects.
   - Format it into JSONL format required for OpenAI fine-tuning (chat-based format with system/user/assistant roles).

2. **Model Training**
   - Upload JSONL training & validation files to OpenAI.
   - Launch fine-tuning job with optional integration to Weights & Biases for real-time experiment tracking.

3. **Evaluation**
   - Retrieve the fine-tuned model.
   - Run predictions on held-out test data.
   - Compare predicted prices to ground truth and log results.

🛠️ **Weights & Biases Integration**
- `wandb` allows visualization and tracking of the fine-tuning run.
- Integration is handled by passing the API key via the OpenAI dashboard (linked to your OpenAI org).
- Use this for comparing hyperparameters, debugging issues, and long-term monitoring of fine-tuning quality.

📁 **Dependencies**
- `openai`, `wandb`, `anthropic`, `huggingface_hub`, `matplotlib`, `dotenv`, and local classes: `Item`, `Tester`

🔗 Useful Docs:
- OpenAI Fine-tuning: https://platform.openai.com/docs/guides/fine-tuning
- Weights & Biases: https://wandb.ai
"""


# ------------------------------------ Package Install Instructions ----------------------------------
# pip install openai wandb huggingface_hub matplotlib python-dotenv


# ------------------------------------ Imports ----------------------------------
import os
import re
import math
import json
import random
import time
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
# anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
# google_api_key = os.getenv('GOOGLE_API_KEY')
# huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

print("Checking API Keys...\n")
if openai_api_key: print(f"OpenAI Key found: {openai_api_key[:10]}...")
# if anthropic_api_key: print(f"Anthropic Key found: {anthropic_api_key[:10]}...")
# if google_api_key: print(f"Google Key found: {google_api_key[:10]}...")
# if huggingface_token: print(f"HuggingFace Token found: {huggingface_token[:10]}...")

print("\n------------------------------------\n")
# Log into Hugging Face (necessary for items.py tokenizer)
# login(huggingface_token, add_to_git_credential=True)
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


# ------------------------------------ Step 1: Prepare Data (Functions)----------------------------------
def messages_for(item):
    """
    This function formats an Item object into a list of messages that can be used for fine-tuning.
    It creates a system message, a user prompt, and an assistant response with the estimated price
    Args:
        item (Item): The Item object containing the price and test prompt.
    Returns:
        list: A list of dictionaries representing the messages in the required format.
    """

    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    # Format the user prompt by removing unnecessary text
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        # The assistant's response is the price formatted to two decimal places.  Feed it, 'Price is $',
        # to try to get the model to respond with just the price.
        {"role": "assistant", "content": f"Price is ${item.price:.2f}"}
    ]


def make_jsonl(items):
    """
    This function takes a list of Item objects and converts them into a JSONL string format
    suitable for fine-tuning with OpenAI. Each item is converted into a JSON object with
    a "messages" key containing a list of messages.
    """
    result = ""
    for item in items:
        messages = messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str +'}\n'
    return result.strip()


def write_jsonl(items, filename):
    """
    This function takes a list of Item objects and writes them to a JSONL file.
    Each item is converted into a JSON object with a "messages" key containing a list of messages.
    Args:
        items (list): A list of Item objects to be converted to JSONL format.               
        filename (str): The name of the file to write the JSONL data to.
    """
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)


# ------------------------------------ Step 1: Prepare Data ----------------------------------
print(f"------- Step 1: Prepare Data -------")

# OpenAI recommends fine-tuning with populations of 50-100 examples.  We are going to use 200 
# examples because our examples are very small
fine_tune_train = train[:200]
fine_tune_validation = train[200:250]

print(f"\nExample message:\n{messages_for(train[0])}")
# Example message: 
# [{'role': 'system', 'content': 'You estimate prices of items. Reply only with the price, no explanation'}, 
# {'role': 'user', 'content': 'How much does this cost?\n\nDelphi FG0166 Fuel Pump Module\nDelphi brings 80 years of OE Heritage into each Delphi pump, ensuring quality and fitment for each Delphi part. Part is validated, tested and matched to the right vehicle application Delphi brings 80 years of OE Heritage into each Delphi assembly, ensuring quality and fitment for each Delphi part Always be sure to check and clean fuel tank to avoid unnecessary returns Rigorous OE-testing ensures the pump can withstand extreme temperatures Brand Delphi, Fit Type Vehicle Specific Fit, Dimensions LxWxH 19.7 x 7.7 x 5.1 inches, Weight 2.2 Pounds, Auto Part Position Unknown, Operation Mode Mechanical, Manufacturer Delphi, Model FUEL PUMP, Dimensions 19.7'}, 
# {'role': 'assistant', 'content': 'Price is $226.95'

# Convert the items into a list of json objects - a "jsonl" string
# Each row represents a message in the form:
# {"messages" : [{"role": "system", "content": "You estimate prices...

print(f"\nExample JSONL:\n{make_jsonl(train[:2])}")
# Example JSONL:
# {"messages": [{"role": "system", "content": "You estimate prices of items. Reply only with the price, no explanation"}, {"role": "user", "content": "How much does this cost?\n\nDelphi FG0166 Fuel Pump Module\nDelphi brings 80 years of OE Heritage into each Delphi pump, ensuring quality and fitment for each Delphi part. Part is validated, tested and matched to the right vehicle application Delphi brings 80 years of OE Heritage into each Delphi assembly, ensuring quality and fitment for each Delphi part Always be sure to check and clean fuel tank to avoid unnecessary returns Rigorous OE-testing ensures the pump can withstand extreme temperatures Brand Delphi, Fit Type Vehicle Specific Fit, Dimensions LxWxH 19.7 x 7.7 x 5.1 inches, Weight 2.2 Pounds, Auto Part Position Unknown, Operation Mode Mechanical, Manufacturer Delphi, Model FUEL PUMP, Dimensions 19.7"}, {"role": "assistant", "content": "Price is $226.95"}]}
# {"messages": [{"role": "system", "content": "You estimate prices of items. Reply only with the price, no explanation"}, {"role": "user", "content": "How much does this cost?\n\nPower Stop Rear Z36 Truck and Tow Brake Kit with Calipers\nThe Power Stop Z36 Truck & Tow Performance brake kit provides the superior stopping power demanded by those who tow boats, haul loads, tackle mountains, lift trucks, and play in the harshest conditions. The brake rotors are drilled to keep temperatures down during extreme braking and slotted to sweep away any debris for constant pad contact. Combined with our Z36 Carbon-Fiber Ceramic performance friction formulation, you can confidently push your rig to the limit and look good doing it with red powder brake calipers. Components are engineered to handle the stress of towing, hauling, mountainous driving, and lifted trucks. Dust-free braking performance. Z36 Carbon-Fiber Ceramic formula provides the extreme braking performance demanded by your truck or 4x"}, {"role": "assistant", "content": "Price is $506.98"}]}

# Write the jsonl string to a file
# The file will be used for fine-tuning the OpenAI model
print(f"\nWriting JSONL files...")
write_jsonl(fine_tune_train, r"C:\Users\Laptop\Desktop\Coding\LLM\Day27\fine_tune_train.jsonl")

# Write the validation set to a separate file       
# This file will be used to evaluate the model's performance during training
write_jsonl(fine_tune_validation, r"C:\Users\Laptop\Desktop\Coding\LLM\Day27\fine_tune_validation.jsonl")


# Load the JSONL files to ensure they are formatted correctly as 'rb' (read binary)
print(f"\nLoading JSONL files...")
with open(r"C:\Users\Laptop\Desktop\Coding\LLM\Day27\fine_tune_train.jsonl", "rb") as f:
    train_file = openai.files.create(file=f, purpose="fine-tune")
# print(train_file)

with open(r"C:\Users\Laptop\Desktop\Coding\LLM\Day27\fine_tune_validation.jsonl", "rb") as f:
    validation_file = openai.files.create(file=f, purpose="fine-tune")
# print(validation_file)


# ------------------------------------ Step 2: Train Model ----------------------------------
# Set up weights and balances (wandb) account: https://wandb.ai (david125tran@gmail.com)
# From the Avatar >> Settings menu, near the bottom, you can create an API key.
# Visit OpenAI dashboard: 
# https://platform.openai.com/account/organization (david112tran@gmail.com)
# in the integrations section, you can add your Weights & Biases (wandb) key.

# This was done using two different emails.  The wandb key is inserted into the OpenAI
# dashboard under david125tran not david112tran. 

# About weights and biases (wandb):
# Weights & Biases (wandb) is a tool for tracking machine learning experiments,
# visualizing results, and sharing findings. It provides a platform to log metrics,
# visualize model performance, and collaborate with team members. It integrates with various
# machine learning frameworks and libraries, making it easy to track experiments and manage datasets.

print(f"\n------- Step 2: Train Model -------")
wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}

if train_file.id: 
    print(f"wandb Key found: {train_file.id[:10]}...")
else:
    print(f"wandb Key not found.")

# Fine-tune the model using the OpenAI API
print(f"\nFine-tuning the model...")
print(f"{openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model="gpt-4o-mini-2024-07-18",
    seed=42,
    hyperparameters={"n_epochs": 1},
    # Add wandb integration for tracking the fine-tuning job
    # This will log the fine-tuning job to Weights & Biases (wandb)
    integrations = [wandb_integration],
    suffix="pricer"
)}")
# FineTuningJob(id='ftjob-gdIpDJE8NdnaQ9NCBNTYZiwK', created_at=1751128122, error=Error(code=None, message=None, param=None), fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(batch_size='auto', learning_rate_multiplier='auto', n_epochs=1), model='gpt-4o-mini-2024-07-18', object='fine_tuning.job', organization_id='org-1263oXvaxwsZiifX4gfI14Ui', result_files=[], seed=42, status='validating_files', trained_tokens=None, training_file='file-EGpAudFmhAXNpi24L5AoRc', validation_file='file-CACbesCbANH9XYioAbW9mc', estimated_finish=None, integrations=[FineTuningJobWandbIntegrationObject(type='wandb', wandb=FineTuningJobWandbIntegration(project='gpt-pricer', entity=None, name=None, tags=None, run_id='ftjob-gdIpDJE8NdnaQ9NCBNTYZiwK'))], metadata=None, method=Method(type='supervised', dpo=None, reinforcement=None, supervised=SupervisedMethod(hyperparameters=SupervisedHyperparameters(batch_size='auto', learning_rate_multiplier='auto', n_epochs=1))), user_provided_suffix='pricer', usage_metrics=None, shared_with_openai=False, eval_id=None)

print(f"{openai.fine_tuning.jobs.list(limit=1)}")
# SyncCursorPage[FineTuningJob](data=[FineTuningJob(id='ftjob-xyWlQQYZLa0GHaWCgcG63kNI', created_at=1751127723, error=Error(code=None, message=None, param=None), fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(batch_size='auto', learning_rate_multiplier='auto', n_epochs=1), model='gpt-4o-mini-2024-07-18', object='fine_tuning.job', organization_id='org-1263oXvaxwsZiifX4gfI14Ui', result_files=[], seed=42, status='validating_files', trained_tokens=None, training_file='file-XxzJBMQ4eoPjzQ99YAUAVe', validation_file='file-Uv1Y3PJKh3v6J2FnPpWELZ', estimated_finish=None, integrations=[FineTuningJobWandbIntegrationObject(type='wandb', wandb=FineTuningJobWandbIntegration(project='gpt-pricer', entity=None, name=None, tags=None, run_id='ftjob-xyWlQQYZLa0GHaWCgcG63kNI'))], metadata=None, method=Method(type='supervised', dpo=None, reinforcement=None, supervised=SupervisedMethod(hyperparameters=SupervisedHyperparameters(batch_size='auto', learning_rate_multiplier='auto', n_epochs=1))), user_provided_suffix='pricer', usage_metrics=None, shared_with_openai=False, eval_id=None)], has_more=False, object='list')

# Get the job ID of the most recent fine-tuning job
# This is used to track the job status and retrieve the fine-tuned model later
print(f"\nGetting the job ID of the most recent fine-tuning job...")
job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id

# Print the job ID
print(f"\nJob ID: {job_id}")
# ftjob-onKEj6cBBaDzf5W48T4tEIWE

# Print the status of the fine-tuning job
print(f"\nChecking the status of the fine-tuning job...")
print(f"\n{openai.fine_tuning.jobs.retrieve(job_id)}")
# FineTuningJob(id='ftjob-onKEj6cBBaDzf5W48T4tEIWE', created_at=1751127916, error=Error(code=None, message=None, param=None), fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(batch_size='auto', learning_rate_multiplier='auto', n_epochs=1), model='gpt-4o-mini-2024-07-18', object='fine_tuning.job', organization_id='org-1263oXvaxwsZiifX4gfI14Ui', result_files=[], seed=42, status='validating_files', trained_tokens=None, training_file='file-KbT4btAtzb9bJsZDnfJy2S', validation_file='file-15hFf7xphbacacwvsVgav8', estimated_finish=None, integrations=[FineTuningJobWandbIntegrationObject(type='wandb', wandb=FineTuningJobWandbIntegration(project='gpt-pricer', entity=None, name=None, tags=None, run_id='ftjob-onKEj6cBBaDzf5W48T4tEIWE'))], metadata=None, method=Method(type='supervised', dpo=None, reinforcement=None, supervised=SupervisedMethod(hyperparameters=SupervisedHyperparameters(batch_size='auto', learning_rate_multiplier='auto', n_epochs=1))), user_provided_suffix='pricer', usage_metrics=None, shared_with_openai=False, eval_id=None)

# Print the events of the fine-tuning job
print(f"\nRetrieving events for the fine-tuning job...")
print(f"{openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data}")


# ------------------------------------ Step 3: Evaluate the Model (Functions) ----------------------------------
def wait_for_model_ready(job_id, check_interval=30):
    """
    This function waits for the fine-tuning job to complete and retrieves the fine-tuned model name.
    It checks the job status every `check_interval` seconds until the model is ready.
    Args:
        job_id (str): The ID of the fine-tuning job to monitor.
        check_interval (int): The number of seconds to wait between status checks.
    Returns:
        str: The name of the fine-tuned model once it is ready.
    """
    while True:
        job = openai.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        model_name = job.fine_tuned_model
        print(f"Status: {status}")
        if model_name:
            print(f"✅ Model is ready: {model_name}")
            return model_name
        time.sleep(check_interval)


def messages_for(item):
    """
    This function formats an Item object into a list of messages that can be used for evaluation.
    It creates a system message, a user prompt, and an assistant response with the estimated price.
    Args:
        item (Item): The Item object containing the price and test prompt.
    Returns:
        list: A list of dictionaries representing the messages in the required format.
    """
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]


def get_price(s):
    """
    This function extracts a price from a string, removing any dollar signs or commas. 
    That the LLM maay have put in.
    """
    s = s.replace('$','').replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0


def gpt_fine_tuned(item):
    """
    This function uses the fine-tuned OpenAI model to estimate the price of an item.
    It sends a request to the OpenAI API with the item's messages and returns the estimated price.
    Args:
        item (Item): The Item object containing the price and test prompt.
    Returns:            
        float: The estimated price of the item as predicted by the fine-tuned model.
    """
    response = openai.chat.completions.create(
        model=fine_tuned_model_name, 
        messages=messages_for(item),
        seed=42,
        max_tokens=7
    )
    reply = response.choices[0].message.content
    return get_price(reply)


# ------------------------------------ Step 3: Evaluate the Model ----------------------------------
print(f"\n------- Step 3: Evaluate the Model -------")

# Wait for the fine-tuning job to complete and retrieve the fine-tuned model name
print(f"\nWaiting for the fine-tuned model to be ready...")
fine_tuned_model_name = wait_for_model_ready(job_id)
print(f"\nFine-tuned model name: {fine_tuned_model_name}")

print(f"message\n: {messages_for(test[0])}")
# message
# [{'role': 'system', 'content': 'You estimate prices of items. Reply only with the price, no explanation'}, {'role': 'user', 'content': "How much does this cost?\n\nOEM AC Compressor w/A/C Repair Kit For Ford F150 F-150 V8 & Lincoln Mark LT 2007 2008 - BuyAutoParts NEW\nAs one of the world's largest automotive parts suppliers, our parts are trusted every day by mechanics and vehicle owners worldwide. This A/C Compressor and Components Kit is manufactured and tested to the strictest OE standards for unparalleled performance. Built for trouble-free ownership and 100% visually inspected and quality tested, this A/C Compressor and Components Kit is backed by our 100% satisfaction guarantee. Guaranteed Exact Fit for easy installation 100% BRAND NEW, premium ISO/TS 16949 quality - tested to meet or exceed OEM specifications Engineered for superior durability, backed by industry-leading unlimited-mileage warranty Included in this K"}, {'role': 'assistant', 'content': 'Price is $'}]

print(f"get_price(): {get_price("The price is roughly $99.99 because blah blah")}")

print(f"\nEvaluating the fine-tuned model on the test set...")
print(f"The actual price: {test[0].price}")
print(f"gpt_fine_tuned price: {gpt_fine_tuned(test[0])}")
# The actual price: 374.41
# gpt_fine_tuned price: 490.0

# Model the performance on the test set
Tester.test(gpt_fine_tuned, test)


