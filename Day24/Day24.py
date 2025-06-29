# ------------------------------------ LLM Data Curation Pipeline ----------------------------------
# This script demonstrates an end-to-end data curation and preparation pipeline for fine-tuning a Large Language Model (LLM).
#
# The project uses real-world e-commerce product data (Amazon Reviews 2023 dataset) to generate high-quality training samples
# for price prediction tasks. It integrates multiple LLM engineering principles including:
#
# 1️⃣ **Data Filtering & Cleaning (items.py)**:
#     - The `Item` class parses raw product metadata, scrubs noisy fields (e.g. part numbers, boilerplate text), enforces 
#       character and token count thresholds, and generates clean text-to-price training prompts suitable for LLM ingestion.
#
# 2️⃣ **Parallelized Dataset Loading (loaders.py)**:
#     - The `ItemLoader` class loads and filters massive datasets efficiently using chunking and multi-process parallelization.
#       Only valid datapoints with meaningful price ranges and sufficient context are retained for further processing.
#
# 3️⃣ **Dataset Balancing & Sampling (Day24.py)**:
#     - This script combines multiple product categories, balances the price distribution, reduces category dominance (e.g. Automotive), 
#       and creates stratified train/test splits for fine-tuning.
#     - The resulting dataset is visualized, analyzed, and optionally pushed to Hugging Face for future model training.
#
# This modular design ensures the LLM is trained on high-quality, domain-specific data while minimizing noise and bias.
# The final dataset enables fine-tuning models like LLaMA 3.1 to learn price estimation from structured product descriptions.


# ------------------------------------ Packages ----------------------------------
# pip install anthropic
# pip install chromadb
# pip install dotenv
# pip install matplotlib


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
    openai_api_key = os.getenv('OPENAI_API_KEY')              # https://openai.com/api/
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')        # https://console.anthropic.com/ 
    google_api_key = os.getenv('GOOGLE_API_KEY')              # https://ai.google.dev/gemini-api
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')         # https://huggingface.co/settings/tokens

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




    # ------------------------------------ Configure Hugging Face Token ----------------------------------
    # Request Access to Llama 3.1-8B:
    # https://huggingface.co/meta-llama/Llama-3.1-8B

    # Log in to the Hugging Face Hub using the retrieved token.
    # `add_to_git_credential=True` stores the token for future Git operations.
    login(huggingface_token, add_to_git_credential=True)

    # This import must come after logging into hugging face or the items.py script breaks.
    # The `Item` class is defined in `items.py` and requires Hugging Face authentication
    from loaders import ItemLoader
    from items import Item

    # Load in the dataset 
    items = ItemLoader("Appliances").load()

    # Look for a familiar item..
    print(items[1].prompt)

    # prints:
    # How much does this cost to the nearest dollar?
    
    # Door Pivot Block - Compatible Kenmore KitchenAid Maytag Whirlpool Refrigerator - Replaces - Quick DIY Repair Solution
    # Pivot Block For Vernicle Mullion Strip On Door - A high-quality exact equivalent for part numbers and Compatibility 
    # with major brands - Door Guide is compatible with Whirlpool, Amana, Dacor, Gaggenau, Hardwick, Jenn-Air, Kenmore, 
    # KitchenAid, and Maytag. Quick DIY repair - Refrigerator Door Guide Pivot Block Replacement will help if your appliance 
    # door doesn't open or close. Wear work gloves to protect your hands during the repair process. Attentive support - If 
    # you are uncertain about whether the block fits your refrigerator, we will help. We generally put forth a valiant effort 
    # to guarantee you are totally

    # Price is $17.00

    # Bring in all of these datasets from the Amazon Reviews 2023 dataset
    # https://huggingface.co/datasets/McAuley-Lab/Amazon
    dataset_names = [
        "Automotive",
        "Electronics",
        "Office_Products",
        "Tools_and_Home_Improvement",
        "Cell_Phones_and_Accessories",
        "Toys_and_Games",
        "Appliances",
        "Musical_Instruments",
    ]

    # Load all the datasets in parallel
    items = []
    for dataset_name in dataset_names:
        loader = ItemLoader(dataset_name)
        items.extend(loader.load())

    print(f"A grand total of {len(items):,} items")
    # A grand total of 2,811,408 items

    # We need to clean up this dataset because it is too large to work with
    # Let's look at the distribution of token counts and prices

    # Plot the distribution of token counts again
    tokens = [item.token_count for item in items]
    plt.figure(figsize=(15, 6))
    plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
    plt.xlabel('Length (tokens)')
    plt.ylabel('Count')
    plt.hist(tokens, rwidth=0.7, color="skyblue", bins=range(0, 300, 10))
    plt.show()

    # Plot the distribution of prices
    prices = [item.price for item in items]
    plt.figure(figsize=(15, 6))
    plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
    plt.show()

    # The token counts are quite high, and the prices are very low
    # Let's see how many items are in each category
    category_counts = Counter()
    for item in items:
        category_counts[item.category]+=1

    # Create a bar chart of the categories
    categories = category_counts.keys()
    counts = [category_counts[category] for category in categories]

    # Bar chart by category from dataset_names list
    plt.figure(figsize=(15, 6))
    plt.bar(categories, counts, color="goldenrod")
    plt.title('How many in each category')
    plt.xlabel('Categories')
    plt.ylabel('Count')

    plt.xticks(rotation=30, ha='right')

    # Add value labels on top of each bar
    for i, v in enumerate(counts):
        plt.text(i, v, f"{v:,}", ha='center', va='bottom')

    # Display the chart
    plt.show()

    # Create a dict with a key of each price from $1 to $999
    # And in the value, put a list of items with that price (to nearest round number)
    # We do this so we can sample evenly across the price range to create a smaller and 
    # more balanced dataset
    slots = defaultdict(list)
    for item in items:
        slots[round(item.price)].append(item)

    # Create a dataset called "sample" which tries to more evenly take from the range of prices
    # And gives more weight to items from categories other than Automotive because Automotive
    # is the most common category and we want to balance the dataset a bit more.  The Automotive
    # dataset is dominating the dataset and we want to reduce its influence to our model.
    # We will take 1200 items from each slot, but if there are less than 1200 items in the slot,
    # we will take all of them.  If there are more than 1200 items in the slot, we will take 1200 items
    # but give more weight to items.

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    sample = []
    # Iterate through the slots and sample items
    for i in range(1, 1000):
        slot = slots[i]
        # If the slot has more than 240 items, we take all of them
        # If the slot has less than 1200 items, we take all of them
        # If the slot has more than 1200 items, we take 1200 items
        # but give more weight to items from the Automotive category
        # We do this to balance the dataset and reduce the influence of the Automotive category
        # We also set a random seed for reproducibility
        if i>=240:
            sample.extend(slot)
        elif len(slot) <= 1200:
            sample.extend(slot)
        else:   
            weights = np.array([1 if item.category=='Automotive' else 5 for item in slot])
            weights = weights / np.sum(weights)
            selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
            selected = [slot[i] for i in selected_indices]
            sample.extend(selected)

    print(f"There are {len(sample):,} items in the sample")

    # Plot the distribution of prices in sample
    prices = [float(item.price) for item in sample]
    plt.figure(figsize=(15, 10))
    plt.title(f"Avg {sum(prices)/len(prices):.2f} and highest {max(prices):,.2f}\n")
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.hist(prices, rwidth=0.7, color="darkblue", bins=range(0, 1000, 10))
    plt.show()

    # We raised the average price and have a smooth population of prices
    # Let's see the categories
    category_counts = Counter()
    for item in sample:
        category_counts[item.category]+=1

    # Create a bar chart of the categories
    categories = category_counts.keys()
    counts = [category_counts[category] for category in categories]

    # Create bar chart
    plt.figure(figsize=(15, 6))
    plt.bar(categories, counts, color="lightgreen")
    plt.title('How many in each category')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')

    # Add value labels on top of each bar
    for i, v in enumerate(counts):
        plt.text(i, v, f"{v:,}", ha='center', va='bottom')

    # Display the chart
    plt.show()

    # Automotive still in the lead, but improved somewhat
    # For another perspective, let's look at a pie

    plt.figure(figsize=(12, 10))
    plt.pie(counts, labels=categories, autopct='%1.0f%%', startangle=90)

    # Add a circle at the center to create a donut chart (optional)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Categories')

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  

    plt.show()

    # How does the price vary with the character count of the prompt?

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
    plt.show()

    # Let's look at a sample item
    # We will look at the 398000th item in the sample
    # This is a random item, but it is in the middle of the sample
    def report(item):
        prompt = item.prompt
        tokens = Item.tokenizer.encode(item.prompt)
        print(prompt)
        print(tokens[-10:])
        print(Item.tokenizer.batch_decode(tokens[-10:]))

    # Let's look at the item
    print(report(sample[398000]))


    random.seed(42)
    random.shuffle(sample)
    train = sample[:400_000]
    test = sample[400_000:402_000]
    print(f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items")

    print(train[0].prompt)

    print(test[0].test_prompt())


    # Plot the distribution of prices in the first 250 test points
    prices = [float(item.price) for item in test[:250]]
    plt.figure(figsize=(15, 6))
    plt.title(f"Avg {sum(prices)/len(prices):.2f} and highest {max(prices):,.2f}\n")
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.hist(prices, rwidth=0.7, color="darkblue", bins=range(0, 1000, 10))
    plt.show()


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


    # Uncomment these lines to push to Hugging Face

    HF_USER = "david125tran"
    DATASET_NAME = f"{HF_USER}/pricer-data"
    dataset.push_to_hub(DATASET_NAME, private=False)

    # One more thing!
    # Let's pickle the training and test dataset so we don't have to execute all this code next time!

    with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day24\train.pkl', 'wb') as file:
        pickle.dump(train, file)

    with open(r'\C:\Users\Laptop\Desktop\Coding\LLM\Day24test.pkl', 'wb') as file:
        pickle.dump(test, file)
