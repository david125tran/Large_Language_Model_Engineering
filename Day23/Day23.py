# ------------------------------------ Packages ----------------------------------
# pip install anthropic
# pip install chromadb
# pip install dotenv
# pip install matplotlib

# ------------------------------------ Imports ----------------------------------
import os
from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
import matplotlib.pyplot as plt


# ------------------------------------ Constants / Variables ----------------------------------


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
from items import Item

# ------------------------------------ Load the Hugging Face Dataset ----------------------------------
# https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

# Loads the "Amazon-Reviews-2023" dataset, specifically the "raw_meta_Appliances" subset.
# `split="full"` loads the entire dataset, and `trust_remote_code=True` allows execution
# of custom code from the dataset repository, which can be necessary for certain datasets.
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_Appliances", split="full", trust_remote_code=True)


# ------------------------------------ Inspect the Data ----------------------------------
# Print the total number of appliance entries in the loaded dataset
print(f"Number of Appliances: {len(dataset):,}")
# Number of Appliances: 94,327

# Investigate a particular datapoint for examination
datapoint = dataset[2]

# Print fields of the selected datapoint to understand its structure and content
print(datapoint["title"])           # Clothes Dryer Drum Slide, General Electric, Hotpoint, WE1M333, WE1M504
print(datapoint["description"])     # ['Brand new dryer drum slide, replaces General Electric, Hotpoint, RCA, WE1M333, WE1M504.']
print(datapoint["features"])        # []
print(datapoint["details"])         # {"Manufacturer": "RPI", "Part Number": "WE1M333,", "Item Weight": "0.352 ounces", "Package Dimensions": "5.5 x 4.7 x 0.4 inches", "Item model number": "WE1M333,", "Is Discontinued By Manufacturer": "No", "Item Package Quantity": "1", "Batteries Included?": "No", "Batteries Required?": "No", "Best Sellers Rank": {"Tools & Home Improvement": 1315213, "Parts & Accessories": 181194}, "Date First Available": "February 25, 2014"}
print(datapoint["price"])           # None

# How many have prices?
# Initialize a counter for datapoints with valid prices.
prices = 0
# Iterate through each datapoint in the dataset.
for datapoint in dataset:
    try:
        # Attempt to convert the 'price' field to a float.
        price = float(datapoint["price"])
        # If the price is positive, increment the counter.
        if price > 0:
            prices += 1
    except ValueError as e:
        # If a ValueError occurs (e.g., price is not a valid number), ignore it.
        pass

# Print the count and percentage of datapoints that have a positive price.
print(f"There are {prices:,} with prices which is {prices/len(dataset)*100:,.1f}%")
# There are 46,726 with prices which is 49.5%

# For those with prices, gather the price and the length:
# Initialize lists to store prices and content lengths.
prices = []
lengths = []

# Iterate through each datapoint again.
for datapoint in dataset:
    try:
        # Attempt to convert the 'price' field to a float.
        price = float(datapoint["price"])
        # If the price is positive, process the datapoint.
        if price > 0:
            prices.append(price) # Add the price to the prices list.
            # Concatenate relevant text fields to create a single content string.
            contents = datapoint["title"] + str(datapoint["description"]) + str(datapoint["features"]) + str(datapoint["details"])
            lengths.append(len(contents)) # Add the length of the content string to the lengths list.
    except ValueError as e:
        # If a ValueError occurs, ignore it.
        pass

# Plot the distribution of lengths:
# Create a new figure for the plot with a specified size.
plt.figure(figsize=(15, 6))
# Set the title of the plot, including average and maximum content lengths.
plt.title(f"Lengths: Avg {sum(lengths)/len(lengths):,.0f} and highest {max(lengths):,}\n")
# Set the label for the x-axis.
plt.xlabel('Length (chars)')
# Set the label for the y-axis.
plt.ylabel('Count')
# Create a histogram of the content lengths.
# `rwidth=0.7` sets the relative width of the bars, `color` sets the bar color,
# and `bins` defines the range and width of the bins.
plt.hist(lengths, rwidth=0.7, color="lightblue", bins=range(0, 6000, 100))
# Display the plot.
plt.show()

# Plot the distribution of prices:
# Create a new figure for the plot with a specified size.
plt.figure(figsize=(15, 6))
# Set the title of the plot, including average and maximum prices.
plt.title(f"Prices: Avg {sum(prices)/len(prices):,.2f} and highest {max(prices):,}\n")
# Set the label for the x-axis.
plt.xlabel('Price ($)')
# Set the label for the y-axis.
plt.ylabel('Count')
# Create a histogram of the prices.
plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
# Display the plot.
plt.show()

# Identify items with exceptionally high prices.
for datapoint in dataset:
    try:
        price = float(datapoint["price"])
        # If the price is greater than 21,000, print its title.
        if price > 21000:
            print(datapoint['title'])   # TurboChef BULLET Rapid Cook Electric Microwave Convection Oven
    except ValueError as e:
        pass

# Create an Item object for each with a price:
# Initialize a list to store `Item` objects.
items = []
# Iterate through each datapoint in the dataset.
for datapoint in dataset:
    try:
        price = float(datapoint["price"])
        # If the price is positive, create an `Item` object.
        if price > 0:
            # Create an Item object encapsulating the raw datapoint and its information for further processing.
            item = Item(datapoint, price)
            # If the `item.include` flag is True (indicating it meets certain criteria), add it to the list.
            if item.include:
                items.append(item)
    except ValueError as e:
        pass

# Print the total number of `Item` objects created.
print(f"There are {len(items):,} items")
# There are 29,191 items

# Look at the first item:
print(items[1])
# <WP67003405 67003405 Door Pivot Block - Compatible Kenmore KitchenAid Maytag Whirlpool Refrigerator - Replaces AP6010352 8208254 PS11743531 - Quick DIY Repair Solution = $16.52>

# Investigate the prompt that will be used during training - the model learns to complete this:
# Print the training prompt for a specific `Item` object (index 100).
# This prompt is what the language model will see as input during training.
print(items[100].prompt)
# Samsung Assembly Ice Maker-Mech
# This is an O.E.M. Authorized part, fits with various Samsung brand models, oem part # this product in manufactured in south Korea. This is an O.E.M. Authorized part Fits with various Samsung brand models Oem part # This is a Samsung replacement part Part Number This is an O.E.M. part Manufacturer J&J International Inc., Part Weight 1 pounds, Dimensions 18 x 12 x 6 inches, model number Is Discontinued No, Color White, Material Acrylonitrile Butadiene Styrene, Quantity 1, Certification Certified frustration-free, Included Components Refrigerator-replacement-parts, Rank Tools & Home Improvement Parts & Accessories 31211, Available April 21, 2011

# Price is $118.00


# Investigate the prompt that will be used during testing - the model has to complete this:
# Print the testing prompt for the same `Item` object.
# This is what the model will receive as input during evaluation or inference.
print(items[100].test_prompt())
# How much does this cost to the nearest dollar?

# Samsung Assembly Ice Maker-Mech
# This is an O.E.M. Authorized part, fits with various Samsung brand models, oem part # this product in manufactured in south Korea. This is an O.E.M. Authorized part Fits with various Samsung brand models Oem part # This is a Samsung replacement part Part Number This is an O.E.M. part Manufacturer J&J International Inc., Part Weight 1 pounds, Dimensions 18 x 12 x 6 inches, model number Is Discontinued No, Color White, Material Acrylonitrile Butadiene Styrene, Quantity 1, Certification Certified frustration-free, Included Components Refrigerator-replacement-parts, Rank Tools & Home Improvement Parts & Accessories 31211, Available April 21, 2011

# Price is $

# Plot the distribution of token counts:
# Extract token counts from all `Item` objects.
tokens = [item.token_count for item in items]
# Create a new figure for the plot.
plt.figure(figsize=(15, 6))
# Set the title of the plot, including average and maximum token counts.
plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}\n")
# Set the label for the x-axis.
plt.xlabel('Length (tokens)')
# Set the label for the y-axis.
plt.ylabel('Count')
# Create a histogram of the token counts.
plt.hist(tokens, rwidth=0.7, color="green", bins=range(0, 300, 10))
# Display the plot.
plt.show()

# Plot the distribution of prices (for the filtered items):
# Extract prices from all `Item` objects.
prices = [item.price for item in items]
# Create a new figure for the plot.
plt.figure(figsize=(15, 6))
# Set the title of the plot, including average and maximum prices.
plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
# Set the label for the x-axis.
plt.xlabel('Price ($)')
# Set the label for the y-axis.
plt.ylabel('Count')
# Create a histogram of the prices.
plt.hist(prices, rwidth=0.7, color="purple", bins=range(0, 300, 10))
# Display the plot.
plt.show()

