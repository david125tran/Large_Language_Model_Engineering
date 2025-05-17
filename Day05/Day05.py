# ------------------------------------ Imports ----------------------------------   
import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI


# ------------------------------------ Load Environment Variables ----------------------------------
# Specify the path to your .env file
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

# print(api_key)


# ------------------------------------ Connect to OpenAPI API Platform ----------------------------------   
# Create an OpenAI instance
MODEL = 'gpt-4o-mini'
openai = OpenAI(api_key=api_key)


# ------------------------------------ Functions ---------------------------------- 
# A class to represent a Webpage

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        """
        Initialize the Website object with a URL and scrape the webpage.
        """
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        # Extract the text from the body of the webpage
        # Remove irrelevant tags
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        # If there is no body, set text to an empty string
        else:
            self.text = ""
        # Extract all links from the webpage
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

def get_links_user_prompt(website):
    user_prompt = ""
    # Extract all links and append links to the user prompt
    user_prompt += "\n".join(website.links)

    return user_prompt

def get_links(url, link_system_prompt):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
      ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

def create_resume(links, link_system_prompt):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": str(links)}
          ],
    )
    result = response.choices[0].message.content

    # Return the response 
    return result


# ------------------------------------ Main Loop ---------------------------------- 
url = "https://github.com/david125tran?tab=repositories"

# Create a Website object
website = Website(url)

# # Print the links
# print("Links found on the webpage:")    
# for link in website.links:
#     print(link)

link_system_prompt = """
You are provided with a list of links found on my Github page where I have a 
portfolio of my projects. Decide which of the links would be most relevant to 
include in a resume detailing my projects to show to a potential employer. You 
should only include links that are relevant projects.  Don't include any links 
that are not relevant to my projects.

You should respond in JSON as in this example:
{
    "links": [
        {"type": "projects": "url": "https://github.com/david125tran/<ProjectName>"},
    ]
}
"""

ai_response = get_links(url, link_system_prompt)

# Print the response    
print("AI Response:")
print("Links found on the webpage:")
print(ai_response)

# Save the response to a JSON file
with open(r"C:\Users\Laptop\Desktop\Coding\LLM\Day05\links.json", "w") as f:
    json.dump(ai_response, f, indent=4)

link_system_prompt = """
You are provided with a list of links of my relevant projects from my Github page where I have 
a portfolio of my projects in JSON format stored in a string.

Example:
{
    "links": [
        {"type": "projects": "url": "https://github.com/david125tran/<ProjectName>"},
    ]
}

Create a detailed markdown resume for me with the links provided to showcase my
projects to show to a potential employer.  Write me a biography showcasing what skills
that I have and what I can do.  Make sure to include the links in the markdown
resume.  Make sure to include the links in the markdown resume.  Add some emojis to
the markdown resume to make it more visually appealing. 
"""

markdown_response = create_resume(ai_response, link_system_prompt)

# Save the response to a markdown file
with open(r"C:\Users\Laptop\Desktop\Coding\LLM\Day05\resume.md", "w", encoding="utf-8") as f:
    f.write(markdown_response)

print("Markdown resume created successfully!")


