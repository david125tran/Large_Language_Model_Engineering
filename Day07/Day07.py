# ------------------------------------ Imports ----------------------------------   
import anthropic
import base64
from dotenv import load_dotenv
import gradio as gr
import google.generativeai
import io
from openai import OpenAI, OpenAIError
import os
from PIL import Image
import requests

# ------------------------------------ Configure API Key ----------------------------------
# https://openai.com/api/


# ------------------------------------ Load Environment Variables ----------------------------------
# Specify the path to the .env file
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

# Access the API keys stored in the environment variable
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')            # https://openai.com/api/
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')      # https://console.anthropic.com/ 
google_api_key = os.getenv('GOOGLE_API_KEY')            # https://ai.google.dev/gemini-api

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


# ------------------------------------ Connect to LLM API Platforms ----------------------------------   
openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()


# ------------------------------------ Functions ----------------------------------  
def message_gpt(prompt):
    system_message = "You are a helpful assistant"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
    )
    return completion.choices[0].message.content

def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()

# ------------------------------------ Connect to OpenAI's API Platform ----------------------------------  
# See how up to date the LLM is.  October 5th 2023.
# print(message_gpt("What is today's date?"))


# ------------------------------------ Create a Simple Gradio Interface Configured to OpenAI's 'gpt-4o-mini' Model ----------------------------------
# Create a Gradio interface running a web server
# force_dark_mode = """
# function refresh() {
#     const url = new URL(window.location);
#     if (url.searchParams.get('__theme') !== 'dark') {
#         url.searchParams.set('__theme', 'dark');
#         window.location.href = url.href;
#     }
# }
# """
# gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never", js=force_dark_mode).launch()


# view = gr.Interface(
#     fn=message_gpt,
#     inputs=[gr.Textbox(label="Your message:", lines=6)],
#     outputs=[gr.Textbox(label="Response:", lines=6)],
#     flagging_mode="never"
# )
# view.launch()



# ------------------------------------ Create a Simple Gradio Interface Configured to OpenAI's 'dall-e-2' Model For Image Editing ----------------------------------
client = OpenAI(api_key=openai_api_key)  


def generate_image(prompt: str, selected_image_path, num_images: int, size: str, quality: str = None) -> list:
    """
    Generates or edits an image using OpenAI DALL·E 2.
    Returns: List of local file paths for Gradio to display.
    """
    try:
        # If there is a selected image, we will edit it
        if selected_image_path:  
            # Read the image file
            img = Image.open(selected_image_path.name).convert("RGB")
            # Resize the image to 1024x1024 for DALL·E editing, DALL·E editing requires 1024x1024 PNG
            img = img.resize((1024, 1024))  
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            # Save the image to the byte array
            img.save(img_byte_arr, format="PNG")
            # Set the file name for the byte array
            img_byte_arr.name = "image.png"  
            # Seek to the beginning of the byte array
            img_byte_arr.seek(0)

            # Create transparent mask
            mask = Image.new("RGBA", img.size, (255, 255, 255, 0))
            # Convert mask to bytes
            mask_byte_arr = io.BytesIO()
            # Save the mask to the byte array
            mask.save(mask_byte_arr, format="PNG")
            # Set the file name for the mask byte array
            mask_byte_arr.name = "mask.png"
            # Seek to the beginning of the mask byte array
            mask_byte_arr.seek(0)

            # Edit the image using DALL·E
            # Note: The mask is used to specify the area to be edited
            response = client.images.edit(
                model="dall-e-2",
                prompt=prompt,
                image=img_byte_arr,
                mask=mask_byte_arr,
                n=num_images,
                size=size,
                response_format="url"
            )   
        else:  # Image Generation
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                n=num_images,
                size=size,
                response_format="url"
            )

        # Extract image URLs from the response
        image_urls = [data.url for data in response.data]

        # Save images locally for Gradio
        save_dir = os.path.join(os.path.expanduser("~"), "Downloads", "GeneratedImages")
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []

        # Download and save each image to the local directory with each having a unique name to prevent overwriting
        for i, image_url in enumerate(image_urls):
            try:
                img_data = requests.get(image_url).content
                file_path = os.path.join(save_dir, f"generated_image_{i+1}.png")
                with open(file_path, "wb") as f:
                    f.write(img_data)
                saved_paths.append(file_path)
            except Exception as e:
                print(f"Error saving image {i+1}: {e}")

        return saved_paths
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def gradio_ui():
    """
    Defines the Gradio UI for generating or editing images using DALL·E 2.
    """
    with gr.Blocks() as iface:
        gr.Markdown("## 🎨 Generate or Edit Images with DALL·E 2")

        with gr.Row():
            prompt_input = gr.Textbox(label="Enter your prompt", placeholder="A futuristic cityscape at sunset")
            image_input = gr.File(label="Optional: Upload image to edit", file_types=["image"])

        with gr.Row():
            num_images_slider = gr.Slider(1, 5, value=1, step=1, label="Number of Images")
            size_dropdown = gr.Dropdown(
                choices=["1024x1024", "512x512", "256x256"],
                value="1024x1024",
                label="Image Size"
            )

        generate_button = gr.Button("Generate Images")
        output_images = gr.Gallery(label="Generated Images", show_label=True)

        generate_button.click(
            fn=generate_image,
            inputs=[prompt_input, image_input, num_images_slider, size_dropdown],
            outputs=output_images
        )

    return iface


# ------------------------------------ Main ----------------------------------
if __name__ == "__main__":
    ui = gradio_ui()
    ui.launch()