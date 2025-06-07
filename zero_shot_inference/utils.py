import os
import base64
import openai
import json
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
openai.api_key = os.getenv("OPENAI_API_KEY")
IMAGE_ROOT = os.getenv("IMAGE_ROOT")

def load_test_file(txt_path, dataset):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    image_paths = []
    for line in lines:
        parts = line.strip().split()
        relative_path = parts[0]
        image_id = os.path.basename(relative_path)
        full_path = os.path.join(IMAGE_ROOT, dataset, relative_path)
        image_paths.append((full_path, image_id))
    return image_paths

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_prompt(prompt_file_path):
    with open(prompt_file_path, "r") as f:
        return f.read().strip()

def call_openai_api(image_base64, prompt, model="gpt-4o-mini"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=100,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def save_prediction(output_path, image_id, prediction):
    with open(output_path, "a") as f:
        json.dump({"image": image_id, "prediction": prediction}, f)
        f.write("\n")
