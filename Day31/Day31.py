# ------------------------------------ Imports ----------------------------------
import os
import random
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import pickle



# ------------------------------------ Main ----------------------------------
# Ensure that the multiprocessing code runs only when this script is executed directly,
# not when it is imported as a module. This avoids issues with process spawning on Windows.
if __name__ == '__main__':


  # ------------------------------------ Configure API Keys / Tokens ----------------------------------
  # Specify the path to the .env file
  env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"

  # Load the .env file
  load_dotenv(dotenv_path=env_path, override=True)

  # Access the API keys stored in the environment variable
  # openai_api_key = os.getenv('OPENAI_API_KEY')              # https://openai.com/api/
  # anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')        # https://console.anthropic.com/ 
  # google_api_key = os.getenv('GOOGLE_API_KEY')              # https://ai.google.dev/gemini-api
  huggingface_token = os.getenv('HUGGINGFACE_TOKEN')          # https://huggingface.co/settings/tokens

  print("Checking API Keys...\n")
  # if openai_api_key:
  #     print(f"OpenAI API Key exists and begins {openai_api_key[:10]}")
  # else:
  #     print("OpenAI API Key not set")
      
  # if anthropic_api_key:
  #     print(f"Anthropic API Key exists and begins {anthropic_api_key[:10]}")
  # else:
  #     print("Anthropic API Key not set")

  # if google_api_key:
  #     print(f"Google API Key exists and begins {google_api_key[:10]}")
  # else:
  #     print("Google API Key not set")

  if huggingface_token:
      print(f"Hugging Face Token exists and begins {huggingface_token[:10]}")
  else:
      print("Hugging Face Token not set")
  print("\n------------------------------------\n")


  # ------------------------------------ Configure Hugging Face Token ----------------------------------
  # Request Access to Llama 3.1-8B:
  # https://huggingface.co/meta-llama/Llama-3.1-8B

  # Log in to the Hugging Face Hub using the retrieved token.
  # `add_to_git_credential=True` stores the token for future Git operations.
  login(huggingface_token, add_to_git_credential=True)

  # This import must come after logging into hugging face or the items.py script breaks.
  # The `Item` class is defined in `items.py` and requires Hugging Face authentication
  from items import Item
  from loaders import ItemLoader


  # ------------------------------------ Load in Data Set ----------------------------------
  # Bring in only the 'Appliances' dataset from the Amazon Reviews 2023 dataset
  # https://huggingface.co/datasets/McAuley-Lab/Amazon
  dataset_names = [
      # "Automotive",
      # "Electronics",
      # "Office_Products",
      # "Tools_and_Home_Improvement",
      # "Cell_Phones_and_Accessories",
      # "Toys_and_Games",
      "Appliances",
      # "Musical_Instruments",
  ]

  # Load all the datasets in parallel
  items = []
  for dataset_name in dataset_names:
      loader = ItemLoader(dataset_name)
      items.extend(loader.load())
    
  print(f"Data set loaded in...")
  print(f"A grand total of {len(items):,} items")
  # A grand total of 28,625 items


  # ------------------------------------ Figure 1 - Token Count Distribution ----------------------------------
  # Plot the distribution of token counts
  tokens = [item.token_count for item in items]
  plt.figure(figsize=(15, 6))
  plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
  plt.xlabel('Length (tokens)')
  plt.ylabel('Count')
  plt.hist(tokens, rwidth=0.7, color="skyblue", bins=range(0, 300, 10))
  # plt.show()


  # ------------------------------------ Figure 2 - Price Distribution ----------------------------------
  # Plot the distribution of prices
  prices = [item.price for item in items]
  plt.figure(figsize=(15, 6))
  plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
  plt.xlabel('Price ($)')
  plt.ylabel('Count')
  plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
  # plt.show()


  # ------------------------------------ Figure 3 - Price and Character Count Correlation ----------------------------------
  # How does the price vary with the character count of the prompt?
  sample = items
  sizes = [len(item.prompt) for item in sample]
  prices = [item.price for item in sample]
  # Create the scatter plot
  plt.figure(figsize=(15, 8))
  plt.scatter(sizes, prices, s=0.2, color="red")
  # Add labels and title
  plt.xlabel('Size')
  plt.ylabel('Price')
  plt.title('Is there a simple correlation?')
  # Display the plot
  # plt.show()
  # The figure shows there is a correlation between price and character count

  # ------------------------------------ Figure 4 - Price Distribution (First 250 Test Points) ----------------------------------
  def report(item):
    """
    Displays information about the prompt and its tokenization for a given item.

    This function prints:
    - The raw prompt text from the item.
    - The last 10 token IDs resulting from tokenizing the prompt.
    - The corresponding decoded tokens for those IDs.

    Args:
        item (Item): An instance of the `Item` class containing a `prompt` attribute.

    Returns:
        None
    """
    prompt = item.prompt
    tokens = Item.tokenizer.encode(item.prompt)
    print(prompt)

    # How much does this cost to the nearest dollar?

    # 285746 OR 285811 Washer Agitator Support And Dogs Compatible with Inglis, Whirlpool, Kenmore, Roper, Admiral
    # 285746 OR 285811 Agitator support and dogs Washing machine agitator repair kit with a medium length cam Agitator support and dogs for two piece agitators.This kit should be used when the top part of the agitator is not moving properly but the bottom part is. Replaces Old Numbers 2744 285746 285811 Washer Agitator Repair Kit. This part works with the following brands Whirlpool, Roper, Admiral, Maytag, Hardwick, Jenn-Air, Estate, Magic Chef, Crosley, Inglis, Norge, Modern Maid, Amana, Kenmore

    # Price is $8.00

    print(tokens[-10:])
    # [11, 14594, 6518, 271, 7117, 374, 400, 23, 13, 410]

    print(Item.tokenizer.batch_decode(tokens[-10:]))
    # [',', ' Ken', 'more', '\n\n', 'Price', ' is', ' $', '8', '.', '00']
    
  print(report(sample[50]))
  # None

  # Shuffle all items using a fixed seed
  random.seed(42)
  random.shuffle(sample)
  
  # 25,000 items will go into training
  train = sample[:25_000]

  # 2,000 items will go into testing
  test = sample[25_000:27_000]

  print(f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items")
  # Divided into a training set of 25,000 items and test set of 2,000 items

  print(train[0].prompt)

  # How much does this cost to the nearest dollar?

  # and Replacement Range Cooktop Drip Pans fit GE, Hotpoint - Two 6 Inch and Two 8 Inch Pans (4 pieces)
  # Contents 2 x (6 inches) and 2 x (8 inches) bowls, 4 drip bowls total Compatibility This replacement kit works with GE, Hotpoint, Moffat, Monogram (GE), Profile (GE), RCA (GE), and Roper models prior to 1996. replaces 65975, replaces and 65974, 770169 Premium quality Drip bowls are made of durable high-quality material. It features a chrome finish, well-tested by the manufacturer. Durable, stick-free, easy to clean, and dishwasher safe. Ensure long-lasting and effective performance Easy to install Shut off electrical power, tilt the coil

  # Price is $12.00

  print(test[0].test_prompt())

  # How much does this cost to the nearest dollar?

  # Setpower Insulated Protective Cover for AJ30 Portable Refrigerator Freezer, suitable for AJ30 Only
  # Insulation & Waterproof well-made insulation could save battery power and improve cooling efficiency by preventing cold air from flowing away. Durable and Foldable with its oxford cloth outer layer, it's durable and protects your portable refrigerator from scratches and dust. Expanded Bag for Accessories two expanded bags on its side, expand space to store the other accessories. Great Ventilation a hollowed design for positions of vents doesn't affect the ventilation. Attention this insulated cover is ONLY suitable for SetPower AJ30 portable refrigerator. FIT TO AJ30 ONLY. Brand Name Setpower, Model Info AJ30 COVER, model number AJ30 COVER, Installation Type Freestanding, Part AJ30 cover, Special Features Portable, Color

  # Price is $

  # Plot the distribution of prices in the first 250 test points
  prices = [float(item.price) for item in test[:250]]
  plt.figure(figsize=(15, 6))
  plt.title(f"Avg {sum(prices)/len(prices):.2f} and highest {max(prices):,.2f}\n")
  plt.xlabel('Price ($)')
  plt.ylabel('Count')
  plt.hist(prices, rwidth=0.7, color="darkblue", bins=range(0, 1000, 10))
  # plt.show()

  # ------------------------------------ Generate the Training and Testing Prompts and Prices Data Sets ----------------------------------
  train_prompts = [item.prompt for item in train]
  train_prices = [item.price for item in train]
  test_prompts = [item.test_prompt() for item in test]
  test_prices = [item.price for item in test]

  # Create a Dataset from the lists
  train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
  test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})

  dataset = DatasetDict({
      "train": train_dataset,
      "test": test_dataset
  })

  # Push to HuggingFace
  DATASET_NAME = "david125tran/lite-data"
  dataset.push_to_hub(DATASET_NAME, private=False)

  # Pickle the training and test datasets 
  with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day31\train_lite.pkl', 'wb') as file:
      pickle.dump(train, file)

  with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day31\test_lite.pkl', 'wb') as file:
      pickle.dump(test, file)

  print("Done!")