# ------------------------------------ LLM Engineering Project: Price Estimation Pipeline ----------------------------------
"""
This script demonstrates a complete end-to-end pipeline for preparing, cleaning, feature extracting,
and building multiple predictive models (including ML models and NLP embeddings) to estimate product prices.

The broader goal is to simulate how Large Language Models (LLMs) like LLaMA 3.1 can be fine-tuned for structured tasks,
using real-world ecommerce data (Amazon Reviews Dataset 2023) with human-readable prompts.

Key LLM Engineering components demonstrated:

1. **Data Curation (via `items.py`)**
    - Scrub noisy ecommerce product data (title, description, features)
    - Apply token and character length constraints to ensure good LLM training examples
    - Format data into training prompts suitable for supervised fine-tuning

2. **Dataset Loading & Preparation**
    - Load pre-curated train/test datasets (`train.pkl` and `test.pkl`)
    - Extract structured features (item weight, brand, text length, etc.) for traditional ML models

3. **Multiple Modeling Strategies**
    - Pure ML Regression models (Linear Regression, SVM, Random Forest)
    - Text-only NLP embeddings via Bag-of-Words (BoW) and Word2Vec
    - Serve as proxies for comparing classical ML models against LLM capabilities

4. **Model Evaluation & Visualization**
    - Custom testing harness (Tester class)
    - RMSLE metric reporting
    - Visualization of predictions vs true prices

In real LLM engineering workflows, this pipeline reflects steps needed before fine-tuning an instruction-tuned LLM
on tabular + text data. The goal is to maximize context signal and minimize noise to help the model learn nuanced price prediction.
"""


# ------------------------------------ Package Install Instructions ----------------------------------
# pip install anthropic chromadb dotenv matplotlib pandas scikit-learn gensim transformers


# ------------------------------------ Imports ----------------------------------
import os
import math
import json
import random
from dotenv import load_dotenv
from huggingface_hub import login
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import Counter

# ML libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

# NLP libraries
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


# ------------------------------------ Console Output Colors ----------------------------------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}


# ------------------------------------ Log In to LLM API Platforms ----------------------------------
# Load API Keys from local .env file (for LLM fine-tuning later)
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"
load_dotenv(dotenv_path=env_path, override=True)

# Print out available API keys for safety check
# openai_api_key = os.getenv('OPENAI_API_KEY')
# anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
# google_api_key = os.getenv('GOOGLE_API_KEY')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

print("Checking API Keys...\n")
# if openai_api_key: print(f"OpenAI Key found: {openai_api_key[:10]}...")
# if anthropic_api_key: print(f"Anthropic Key found: {anthropic_api_key[:10]}...")
# if google_api_key: print(f"Google Key found: {google_api_key[:10]}...")
if huggingface_token: print(f"HuggingFace Token found: {huggingface_token[:10]}...")

print("\n------------------------------------\n")
# Log into Hugging Face (necessary for items.py tokenizer)
login(huggingface_token, add_to_git_credential=True)
# Import Item class AFTER huggingface login (due to tokenizer auth)
from items import Item



# ------------------------------------ Extract *.pkl Files ----------------------------------
# Load previously pre-processed datasets (pickle files)
with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day25\train.pkl', 'rb') as file:
    train = pickle.load(file)
with open(r'C:\Users\Laptop\Desktop\Coding\LLM\Day25\test.pkl', 'rb') as file:
    test = pickle.load(file)

# Right after loading the datasets, we can take the JSON object blobs and parse them into structured features.
# Each Item object has a 'details' field which is a JSON string containing structured metadata.
# This metadata includes things like brand, dimensions, weight, and other product attributes.
# We will parse this JSON into a Python dictionary for easier access to features.
# This allows us to extract structured features for both training and testing datasets.   
for item in train:
    item.features = json.loads(item.details)
for item in test:
    item.features = json.loads(item.details)

# Quick sanity check - show example prompt structures
print("\nExample training prompt:")
print(train[0].prompt)
"""
How much does this cost to the nearest dollar?
Delphi brings 80 years of OE Heritage into each Delphi pump, ensuring quality and fitment for each Delphi part. 
Part is validated, tested and matched to the right vehicle application Delphi brings 80 years of OE Heritage 
into each Delphi assembly, ensuring quality and fitment for each Delphi part Always be sure to check and clean 
fuel tank to avoid unnecessary returns Rigorous OE-testing ensures the pump can withstand extreme temperatures 
Brand Delphi, Fit Type Vehicle Specific Fit, Dimensions LxWxH 19.7 x 7.7 x 5.1 inches, Weight 2.2 Pounds, Auto 
Part Position Unknown, Operation Mode Mechanical, Manufacturer Delphi, Model FUEL PUMP, Dimensions 19.7
Price is $227.00
"""
print("\nExample test prompt:")
print(test[0].prompt)
"""
OEM AC Compressor w/A/C Repair Kit For Ford F150 F-150 V8 & Lincoln Mark LT 2007 2008 - BuyAutoParts NEW
As one of the world's largest automotive parts suppliers, our parts are trusted every day by mechanics and 
vehicle owners worldwide. This A/C Compressor and Components Kit is manufactured and tested to the strictest 
OE standards for unparalleled performance. Built for trouble-free ownership and 100% visually inspected and 
quality tested, this A/C Compressor and Components Kit is backed by our 100% satisfaction guarantee. Guaranteed 
Exact Fit for easy installation 100% BRAND NEW, premium ISO/TS 16949 quality - tested to meet or exceed OEM 
specifications Engineered for superior durability, backed by industry-leading unlimited-mileage warranty 
Included in this K
Price is $374.00
"""


# ------------------------------------ Evaluation Framework (Tester Class) ----------------------------------
class Tester:
    """
    The Tester class is a universal evaluation and benchmarking tool that you use to:
    - Feed in any kind of price predictor function you want to test.
    - Automatically run that predictor on a bunch of test samples.
    - Collect statistics: error, RMSLE (log loss), hit rates.
    - Print out nice colored console logs (green / yellow / red).
    - Plot results visually (scatter plot of predicted vs actual prices).
    """
    def __init__(self, predictor, title=None, data=test, size=250):
        # predictor: any function you pass that can take in an Item and output a price prediction.
        self.predictor = predictor
        # data: defaults to your test set.
        self.data = data
        # title: optional title for the run.
        self.title = title or predictor.__name__.replace("_", " ").title()
        # size: how many test samples to evaluate on (default: 250).
        self.size = size
        # Initialize lists to store results
        self.guesses, self.truths, self.errors, self.sles, self.colors = [],[],[],[],[]

    def color_for(self, error, truth):
        """Color coding for easy console visualization"""
        if error<40 or error/truth < 0.2: return "green"
        if error<80 or error/truth < 0.4: return "orange"
        return "red"

    def run_datapoint(self, i):
        """
        For each datapoint, run the predictor, calculate absolute error (how far off your guess
        is), log error squared (RMSLE), and color code the output based on how big the error is.

        Stores all results in lists for later reporting.  Prints nicely formatted output to console.
        """
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
        print(f"{COLOR_MAP[color]}{i+1}: Guess ${guess:.2f} | Truth ${truth:.2f} | Error ${error:.2f} | SLE {sle:.2f} | Item: {title}{RESET}")

    def chart(self, title):
        """
        Simple scatter plot to visualize the predictions vs true prices. 
        - x-axis: Ground Truth (actual price)
        - y-axis: Model Estimate (predicted price)
        - Diagonal line indicates perfect predictions (y=x).
        - Points are colored based on error size: green (good), yellow (moderate), red (poor).
        """
        max_val = max(max(self.truths), max(self.guesses))
        plt.figure(figsize=(12, 8))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.title(title)
        plt.show()

    def report(self):
        """
        Once all datapoints are processed, calculate overall statistics:
        - Average error across all predictions
        - Root Mean Squared Logarithmic Error (RMSLE)
        - Hit rate (percentage of predictions within acceptable error bounds)
        - Generate a final chart visualizing the predictions vs true prices.
        """
        # Calculate average error
        avg_error = sum(self.errors)/self.size
        # Calculate RMSLE
        rmsle = math.sqrt(sum(self.sles)/self.size)
        # Calculate hit rate (percentage of green predictions)
        hits = sum(1 for color in self.colors if color=="green")
        # Summary statistics
        title = f"{self.title} | Error=${avg_error:.2f} | RMSLE={rmsle:.2f} | Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        """
        For each item in the dataset, run the predictor function, collect results, and print a report.
        This method iterates through the specified number of datapoints (default: 250) and
        calls run_datapoint for each one to gather predictions and errors.
        """
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function):
        """
        A convenience method to quickly test any price predictor function.  You simply call Tester.test(random_pricer)
        and it will run the predictor on the test dataset, print results, and visualize the predictions
        """
        cls(function).run()

# ------------------------------------ Dummy Baseline ----------------------------------
def random_pricer(item):
    """
    Simulates a random price prediction for an item.  That returns a random integer between 1 and 1000.
    It simply returns a random price within a specified range.
    """
    return random.randrange(1, 1000)

# Reset Python’s random number generator to always produce the same sequence of random numbers
random.seed(42)

# Test the 'random_pricer' function using the Tester class
print("\nTesting Random Pricer Baseline:")
Tester.test(random_pricer)


# ------------------------------------ Constant Baseline ----------------------------------
# Calculate the average price from the training set to use as a constant baseline
# This is a simple baseline that predicts the average price of items in the training set.
print("\nTesting Constant Pricer Baseline:")
# Extract prices from the training set and compute the average
training_prices = [item.price for item in train]
training_average = sum(training_prices) / len(training_prices)

def constant_pricer(item):
    """
    This function returns the average price from the training set.
    """
    return training_average

# Test the 'constant_pricer' function using the Tester class
Tester.test(constant_pricer)


# ------------------------------------ Structured Feature Engineering ----------------------------------
# Structured Feature Engineering:
# - Process of transforming raw structured metadata into clean, numeric ML features.
# - In this code: raw item.features contain messy strings, mixed units, dicts, and categories.
# - Feature engineering converts these into model-ready inputs:
#     * Normalize units (weight → pounds)
#     * Aggregate nested dicts (average Best Sellers Rank)
#     * Extract metadata (text prompt length)
#     * Encode categories (top brand → binary flag)
# - Also handles missing values via imputation (using training averages).
# - Converts heterogeneous data into consistent, meaningful signals for ML model.

# Look at 20 most common features in the training set
feature_count = Counter()
for item in train:
    for f in item.features.keys():
        feature_count[f]+=1

print("\nMost common features in training set:")
print(feature_count.most_common(40))

# Extract structured metadata to build ML features (weight, rank, text length, brand type)
def get_weight(item):
    """
    Extracts the weight from the item's features and converts it to pounds.
    The weight is expected to be in a string format like "2.2 pounds" or "100 grams".
    It handles various units like pounds, ounces, grams, milligrams, kilograms, and hundredths.
    Returns the weight in pounds as a float, or None if not available
    """
    weight_str = item.features.get('Item Weight')
    if weight_str:
        parts = weight_str.split(' ')
        amount = float(parts[0])
        unit = parts[1].lower()
        conversions = {
            "pounds": 1, "ounces": 1/16, "grams": 1/453.592, "milligrams": 1/453592,
            "kilograms": 1/0.453592, "hundredths": 1/100
        }
        return amount * conversions.get(unit, 1)
    return None

# Precompute average weight & rank to fill missing values
weights = [get_weight(t) for t in train if get_weight(t)]
avg_weight = sum(weights)/len(weights)

def get_weight_with_default(item):
    """ Returns the weight of the item in pounds, or the average weight if not available. """
    return get_weight(item) or avg_weight

def get_rank(item):
    """ Extracts the average Best Sellers Rank from the item's features. """
    rank_dict = item.features.get("Best Sellers Rank")
    if rank_dict:
        return sum(rank_dict.values()) / len(rank_dict)
    return None

ranks = [get_rank(t) for t in train if get_rank(t)]
avg_rank = sum(ranks)/len(ranks)

def get_rank_with_default(item):
    """ Returns the average Best Sellers Rank of the item, or the average rank if not available. """
    return get_rank(item) or avg_rank

def get_text_length(item):
    """ Returns the length of the item's test prompt. """
    return len(item.test_prompt())

def is_top_electronics_brand(item):
    """ Checks if the item's brand is one of the top electronics brands. """
    brand = item.features.get("Brand")
    # Manually define a list of top electronics brands
    top_brands = ["hp", "dell", "lenovo", "samsung", "asus", "sony", "canon", "apple", "intel"]
    return brand and brand.lower() in top_brands

def get_features(item):
    """ 
    Extracts structured features from an item for model training.

    Returns a dictionary with the following features:
    - weight: Weight of the item in pounds (default to average if missing)  
    - rank: Average Best Sellers Rank (default to average if missing)
    - text_length: Length of the item's test prompt
    - is_top_electronics_brand: Binary indicator if the brand is a top electronics brand
    """
    return {
        "weight": get_weight_with_default(item),
        "rank": get_rank_with_default(item),
        "text_length": get_text_length(item),
        "is_top_electronics_brand": 1 if is_top_electronics_brand(item) else 0
    }

# Create full feature dataframe 
train_df = pd.DataFrame([get_features(item) for item in train])
train_df['price'] = [item.price for item in train]

# Create test feature dataframe (only first 250 items for speed)
test_df = pd.DataFrame([get_features(item) for item in test[:250]])
test_df['price'] = [item.price for item in test[:250]]


# ------------------------------------ Linear Regression ----------------------------------
feature_columns = ['weight', 'rank', 'text_length', 'is_top_electronics_brand']
X_train, y_train = train_df[feature_columns], train_df['price']
X_test, y_test = test_df[feature_columns], test_df['price']

model = LinearRegression().fit(X_train, y_train)

def linear_regression_pricer(item):
    features = pd.DataFrame([get_features(item)])
    return model.predict(features)[0]

Tester.test(linear_regression_pricer)

# ------------------------------------ Bag-of-Words + Linear Regression ----------------------------------
# Bag-of-Words (BoW) is a simple text representation technique that converts text into a matrix of token counts.
# It counts the frequency of each word in the text, ignoring grammar and word order.
# This allows us to create a numerical representation of text data that can be used in machine learning models.
# In this code, we use BoW to convert the item descriptions into a matrix of word counts,
# which we then use as features for a linear regression model to predict prices.
# We use the CountVectorizer from scikit-learn to create the BoW representation.
# We limit the vocabulary to the top 1000 words and remove common English stop words.
# This helps reduce noise and focuses on the most relevant words for price prediction.
# The resulting matrix is then used to train a linear regression model to predict item prices based on their descriptions.

# Our documents are the test prompts from the training set, and we will use these to create a Bag-of-Words representation.
documents = [item.test_prompt() for item in train]

# Extract prices from the training set to use as target values for regression
prices = np.array([item.price for item in train])

# Create a Bag-of-Words representation of the documents using CountVectorizer
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_bow = vectorizer.fit_transform(documents)
regressor = LinearRegression().fit(X_bow, prices)

def bow_lr_pricer(item):
    x = vectorizer.transform([item.test_prompt()])
    return max(regressor.predict(x)[0], 0)

Tester.test(bow_lr_pricer)

# ------------------------------------ Word2Vec Embeddings + Linear Regression ----------------------------------
processed_docs = [simple_preprocess(doc) for doc in documents]

# Build vector with 400 dimensions, using a window size of 5 and minimum word count of 1
# This means we will create word vectors for all words in the documents, even if they appear only once.
# The window size of 5 means that the model will consider a context of 5 words on either side of the target word.
# The vector size of 400 means that each word will be represented by a 400-dimensional vector.
# The workers parameter allows us to use multiple CPU cores for training the model, speeding up the process.
# The Word2Vec model will learn word embeddings from the processed documents.
# These embeddings capture semantic relationships between words based on their context
w2v_model = Word2Vec(sentences=processed_docs, vector_size=400, window=5, min_count=1, workers=8)

def document_vector(doc):
    vectors = [w2v_model.wv[word] for word in simple_preprocess(doc) if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

X_w2v = np.array([document_vector(doc) for doc in documents])
w2v_lr_model = LinearRegression().fit(X_w2v, prices)

def word2vec_lr_pricer(item):
    vec = document_vector(item.test_prompt())
    return max(w2v_lr_model.predict([vec])[0], 0)

Tester.test(word2vec_lr_pricer)

# ------------------------------------ SVM Regression on Word2Vec ----------------------------------
# Support Vector Regression (SVR) is a type of Support Vector Machine (SVM) that is used for regression tasks.
# It works by finding a hyperplane that best fits the data points in a high-dimensional space
# In this code, we use the Word2Vec embeddings as features for the SVR model.
# The Word2Vec embeddings are dense vector representations of the item descriptions, capturing semantic relationships.
# We use the LinearSVR implementation from scikit-learn, which is a linear support vector regression model.
# The model is trained on the Word2Vec embeddings of the training set,
# and then we can use it to predict prices for new items based on their descriptions.
svr_model = LinearSVR().fit(X_w2v, prices)

def svr_pricer(item):
    vec = document_vector(item.test_prompt())
    return max(float(svr_model.predict([vec])[0]), 0)

Tester.test(svr_pricer)

# ------------------------------------ Random Forest Regression on Word2Vec ----------------------------------
# Random Forest Regression is an ensemble learning method that uses multiple decision trees to make predictions.
# It works by training multiple decision trees on different subsets of the data and averaging their predictions.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8).fit(X_w2v, prices)

def random_forest_pricer(item):
    vec = document_vector(item.test_prompt())
    return max(rf_model.predict([vec])[0], 0)

Tester.test(random_forest_pricer)
