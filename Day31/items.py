"""
This Item class is a core data preparation utility for fine-tuning or evaluating an LLM (like Meta 
Llama 3.1) on price prediction tasks. It takes raw product data (title, description, features, details), 
cleans and scrubs it to remove irrelevant noise, filters out uninformative samples based on character 
and token length thresholds, and formats the cleaned text into prompt-response pairs suitable for 
supervised fine-tuning. The class uses a tokenizer from Hugging Face's transformers library to ensure 
the text fits within token limits appropriate for LLM input.

In LLM engineering, this class handles several critical preprocessing steps: noise reduction (removing 
part numbers, boilerplate text), length control (token/character constraints), and structured prompt 
creation for both training (with answers) and inference (without answers). This ensures high-quality, 
consistent, and efficiently tokenized data that aligns well with the expected format the LLM was 
pretrained on, enabling better downstream learning for specialized tasks like product price estimation.

"""

from typing import Optional
from transformers import AutoTokenizer
import re

# --- LLM Configuration Constants ---
# Define the base model for which the tokenizer is being loaded.
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

# --- Tokenization and Character Length Constraints ---
# Minimum number of tokens required for an item to be considered useful for the LLM.
# Items with fewer tokens might lack sufficient context for price prediction.
MIN_TOKENS = 150 

# Maximum number of tokens an item's content should have.
# Content will be truncated to this limit to manage input size for the LLM.
MAX_TOKENS = 160 
# After truncation and adding prompt text, the total token count is estimated to be around 180.

# Minimum number of characters for an item's content.
# Items shorter than this might not provide enough descriptive information.
MIN_CHARS = 300

# A character ceiling based on the maximum tokens. This is a rough estimate (MAX_TOKENS * avg_chars_per_token).
# Used to quickly filter out extremely long text before tokenization, saving computation.
CEILING_CHARS = MAX_TOKENS * 7

class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price, specifically structured for LLM input.
    This class handles the transformation of raw product data into a format suitable for training and testing
    a language model to predict product prices.
    """
    
    # --- Class-level Attributes ---
    # Load the tokenizer associated with the specified base model.
    # `trust_remote_code=True` is necessary for some tokenizers that include custom code.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Prefix string used to indicate the price in the prompt.
    PREFIX = "Price is $"

    # The question posed to the LLM to elicit a price prediction.
    QUESTION = "How much does this cost to the nearest dollar?"
    
    # List of common, uninformative strings to be removed from product details.
    # These often appear in product descriptions but don't add value for price prediction.
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    # --- Instance Attributes ---
    title: str                                      # The title of the product
    price: float                                    # The numerical price of the product
    category: str                                   # The category of the product 
    token_count: int = 0                            # Stores the token count of the final generated prompt. Initialized to 0.
    details: Optional[str]                          # Stores the processed details string, if available. Optional as it might be empty.
    prompt: Optional[str] = None                    # The generated prompt string for the LLM. Initialized to None.
    include = False                                 # A flag indicating whether this Item should be included in the dataset for the LLM. Initialized to False.

    def __init__(self, data, price):
        """
        Initializes an Item object.
        
        Args:
            data (dict): A dictionary containing raw product information (e.g., title, description, details).
            price (float): The price of the product.
        """
        self.title = data['title']                  # Assign the product title from the raw data.
        self.price = price                          # Assign the product price.
        self.parse(data)                            # Call the parse method to process and clean the data.

    def scrub_details(self):
        """
        Cleans up the `details` string by removing predefined common, uninformative phrases.
        This helps reduce noise and keeps the input relevant for the LLM.
        """
        details = self.details                      # Get the current details string.
                                                    # Iterate through the list of REMOVALS and replace them with an empty string.
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details                              # Return the cleaned details string.

    def scrub(self, stuff):
        """
        Performs general cleaning on a given text string.
        This includes:
        - Removing various punctuation and extra whitespace.
        - Removing potentially irrelevant product numbers (words longer than 7 characters containing digits).
        
        Args:
            stuff (str): The text string to be cleaned (e.g., title, description).
        
        Returns:
            str: The cleaned text string.
        """
        # Replace multiple occurrences of specific punctuation and whitespace with a single space.
        # Then, remove leading/trailing whitespace.
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        # Clean up comma spacing and multiple commas.
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(' ') # Split the string into individual words.
        # Filter out "useless part numbers" which are often long and contain digits,
        # as these can inflate token count without adding meaning.
        select = [word for word in words if len(word) < 7 or not any(char.isdigit() for char in word)]
        return " ".join(select) # Join the filtered words back into a single string.
    
    def parse(self, data):
        """
        Parses and processes the raw product data to create a clean, tokenized input
        string for the LLM. It also determines if the item meets the criteria
        (minimum character length, token count) to be included in the dataset.
        
        Args:
            data (dict): The raw product data dictionary.
        """
        # Concatenate description lines, ensuring it's a string.
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n' # Add a newline if description exists.
        
        # Concatenate features lines, ensuring it's a string.
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n' # Add features to contents if they exist.
        
        self.details = data['details'] # Assign raw details.
        if self.details:
            contents += self.scrub_details() + '\n' # Scrub and add details to contents if they exist.
        
        # Check if the total content length exceeds the minimum character threshold.
        if len(contents) > MIN_CHARS:
            # Truncate content to the character ceiling to prevent excessively long inputs.
            contents = contents[:CEILING_CHARS]
            # Combine the scrubbed title and content.
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            # Tokenize the combined text without adding special tokens (like <s>, </s>).
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Check if the token count is above the minimum required.
            if len(tokens) > MIN_TOKENS:
                # Truncate tokens to the maximum allowed token count.
                tokens = tokens[:MAX_TOKENS]
                # Decode the tokens back into a string.
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text) # Create the training prompt using the processed text.
                self.include = True # Mark the item for inclusion as it meets all criteria.

    def make_prompt(self, text):
        """
        Constructs the full prompt string for training the LLM.
        This prompt includes the question, the product text, and the correct price (answer).
        
        Args:
            text (str): The cleaned and tokenized product description text.
        """
        # Format the prompt with the question, product text, and the expected answer (price).
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00" # Append the price in the desired format.
        # Calculate and store the total token count of the complete prompt.
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Generates a prompt suitable for testing or inference, where the actual price
        is intentionally omitted from the prompt. The LLM will then attempt to predict this price.
        """
        # Split the full training prompt at the 'Price is $' prefix and take the first part,
        # then re-add the prefix. This effectively removes the ground truth price.
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Provides a user-friendly string representation of the Item object.
        This is useful for debugging and printing item details.
        """
        return f"<{self.title} = ${self.price}>"