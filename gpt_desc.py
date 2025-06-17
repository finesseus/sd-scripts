import datetime

import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv


# Set your OpenAI API key (via environment variable for security)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def gpt_instructions(user_input: str) -> str:
    instructions = (
        "You are an expert fashion stylist and product photographer.\n"
        "Given any input (from minimal to very detailed), "
        "write a complete, visually rich product photography prompt describing a single top, bottom, dress, outerwear or combination with the following:\n\n"
        "- Describe: garment type, silhouette, fabric, construction, color, styling features "
        "(e.g., neckline, sleeves, skirt shape), and details like hardware or embellishment.\n"
        "- Include: tailoring/texture features (e.g., sheen, mesh, boning, topstitching, ruching).\n"
        "- Include styling details such as straps, cutouts, hardware, zippers, visible seams, etc.\n"
        "- Emphasize: a **trendy, bold, individualistic, Gen-Z aesthetic** (unless otherwise specified).\n"
        "- End with: “Even, professional lighting "
        "to highlight the texture and contours of the garment.”\n\n"
        "If input is vague (like “dress”), creatively infer a complete unique piece that feels fresh and modern. The more specific the user is the less should be inferred\n\n"
        "Example Input: 'green asymmetrical dress with jewels'\n"
        "Example Output: An electric green asymmetrical mini dress made of glossy stretch satin. "
        "The design features a single shoulder strap with chunky rhinestone hardware, a dramatic cutout at the waist, "
        "and an uneven hemline edged in microcrystals. The bodice is sharply structured with darted seams and side boning, "
        "while the skirt hugs the hips and flares slightly. The look balances glam with edge, channeling Gen-Z red carpet energy. "
        "Even, professional lighting to highlight the texture and contours of the garment.\n\n"
        f"Now write the full product prompt for this:\n{user_input}"
    )
    return instructions


def flux_instructions(user_input: str) -> str:
    instructions = (
        "You are an expert fashion stylist and product photographer.\n"
        "Given any input (from minimal to very detailed), "
        "write a complete, visually rich product photography prompt describing a single top, bottom, dress, outerwear or combination with the following:\n\n"
        "- Start with: “A studio product shot of...”.\n"
        "- Describe: garment type, silhouette, fabric, construction, color, styling features "
        "(e.g., neckline, sleeves, skirt shape), and details like hardware or embellishment.\n"
        "- Include: tailoring/texture features (e.g., sheen, mesh, boning, topstitching, ruching).\n"
        "- Include styling details such as straps, cutouts, hardware, zippers, visible seams, etc.\n"
        "- Emphasize: a **trendy, bold, individualistic, Gen-Z aesthetic** (unless otherwise specified).\n"
        "- End with: “Set against a neutral light grey background with even, professional lighting "
        "to highlight the texture and contours of the garment.”\n\n"
        "If input is vague (like “dress”), creatively infer a complete unique piece that feels fresh and modern. The more specific the user is the less should be inferred\n\n"
        "Example Input: 'green asymmetrical dress with jewels'\n"
        "Example Output: An electric green asymmetrical mini dress made of glossy stretch satin. "
        "The design features a single shoulder strap with chunky rhinestone hardware, a dramatic cutout at the waist, "
        "and an uneven hemline edged in microcrystals. The bodice is sharply structured with darted seams and side boning, "
        "while the skirt hugs the hips and flares slightly. The look balances glam with edge, channeling Gen-Z red carpet energy. "
        "Even, professional lighting to highlight the texture and contours of the garment.\n\n"
        f"Now write the full product prompt for this:\n{user_input}"
    )
    return instructions


def generate_fashion_prompt(user_input: str, model: str) -> str:
    if model == "finesse.fl.dev":
        instructions = flux_instructions(user_input)
    else:
        instructions = gpt_instructions(user_input)

    response = client.responses.create(model="gpt-4.1", input=instructions, temperature=0.7, max_output_tokens=500)

    return response.output_text.strip()


def generate_image_2(desc: str):
    MODEL_NAME = "gpt-image-1"
    N_IMAGES = 1
    IMAGE_SIZE = "1024x1536"  # options: "1024x1024","1024x1536","1536x1024","auto"
    QUALITY = "medium"  # choose from "low", "medium", "high"
    BASE_PROMPT = """
    You are a product photographer AI. I will provide you with a description of a piece of clothing.
    Create a product shot of this item or set where the entire piece is in view.
    - If the item is clothing, it should appear on an invisible hourglass figure (no visible mannequin) Make sure all elements of the item is in view.
    - If the item is an accessory, position it as in a studio product shot.
    - The shot should emulate real photography under studio lighting.
    Description:
    {description}
    """

    prompt = BASE_PROMPT.format(description=desc)
    result = client.images.generate(
        model=MODEL_NAME,
        prompt=prompt,
        n=N_IMAGES,
        size=IMAGE_SIZE,
        background="transparent",
        quality=QUALITY,
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Convert to Pillow image
    image = Image.open(BytesIO(image_bytes))
    output_dir = "gpt_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    image.save(output_path)
    return image


def generate_image_from_prompt(prompt: str):
    result = client.images.generate(model="gpt-image-1", prompt=prompt)

    # Decode the base64 image
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Convert to Pillow image
    image = Image.open(BytesIO(image_bytes))
    output_dir = "gpt_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    image.save(output_path)
    return image


if __name__ == "__main__":
    user_input = input("Enter a garment description: ")
    prompt = generate_fashion_prompt(user_input)
    print("\n--- Generated Photography Prompt ---\n")
    print(prompt)
    print("\n--- Generating Image ---\n")
    # generate_image_from_prompt(prompt)
    generate_image_2(prompt)
    print("Image generated and saved successfully.")
