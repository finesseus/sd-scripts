#!/usr/bin/env python3
import os
import time
import base64
from dotenv import load_dotenv
import requests

#!/usr/bin/env python3
import json
import random
from datetime import date
from openai import OpenAI

load_dotenv()
# === CONFIGURE HERE ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable
MODEL_NAME = "gpt-image-1"
N_IMAGES = 2
IMAGE_SIZE = "1024x1536"  # options: "1024x1024","1024x1536","1536x1024","auto"
QUALITY = "medium"  # choose from "low", "medium", "high"
JSON_FILE = "0514/jsons/cluster_design_descriptions_bikinis.json"  # path to your single JSON file
OUTPUT_FOLDER = "images/0514/from_descriptions_swim"
BASE_PROMPT = """
You are a product photographer AI. I will provide you with a description of a piece of clothing.
Create a product shot of this item or set where the entire piece is in view.
- If the item is clothing, it should appear on an invisible hourglass figure (no visible mannequin) Make sure all elements of the item is in view.
- If the item is an accessory, position it as in a studio product shot.
- The shot should emulate real photography under studio lighting.
Description:
{description}
"""


def sanitize_for_filename(s: str) -> str:
    """
    Sanitize a string to be safe for filenames/folders (remove or replace invalid characters).
    """
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def load_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Prepare client and output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Load all descriptions from the single JSON file
    descriptions = load_json(JSON_FILE)
    for key, desc in descriptions.items():
        safe_key = sanitize_for_filename(key)
        prompt = BASE_PROMPT.format(description=desc)
        print(f"Generating {N_IMAGES} images for '{key}'...")
        # Request images in one batch
        response = client.images.generate(
            model=MODEL_NAME,
            prompt=prompt,
            n=N_IMAGES,
            size=IMAGE_SIZE,
            background="transparent",
            quality=QUALITY,
        )
        # Save each generated image into its versioned folder
        for idx, img in enumerate(response.data, start=1):
            # Decode image bytes
            if hasattr(img, "b64_json") and img.b64_json:
                image_bytes = base64.b64decode(img.b64_json)
            elif hasattr(img, "url") and img.url:
                resp = requests.get(img.url)
                resp.raise_for_status()
                image_bytes = resp.content
            else:
                print(f":warning: No image data for {key}, version {idx}")
                continue
            # Create folder per key/version
            # version_folder = os.path.join(OUTPUT_FOLDER, f"{safe_key}_v{idx}")
            # os.makedirs(version_folder, exist_ok=True)
            # Write out PNG
            out_path = os.path.join(OUTPUT_FOLDER, f"{safe_key}_v{idx}.png")
            with open(out_path, "wb") as f_img:
                f_img.write(image_bytes)
            print(f":heavy_check_mark: Saved image: {out_path}")
        # Pause briefly to respect rate limits
        time.sleep(1)
    print("All images generated.")


if __name__ == "__main__":
    main()
