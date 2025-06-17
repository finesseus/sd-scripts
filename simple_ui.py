import gradio as gr
from PIL import Image

from flux_minimal_inference import flux_minimal
from gpt_desc import generate_fashion_prompt, generate_image_from_prompt

# Import or define your model functions
# from diffusion_model import flux_minimal, generate_image_from_prompt, generate_fashion_prompt


def run_flux_minimal(prompt: str) -> Image.Image:
    return flux_minimal(prompt)


def run_gpt(prompt: str) -> Image.Image:
    return generate_image_from_prompt(prompt)


def gradio_generate(prompt: str, model: str):
    # Enhance the prompt for fashion scenarios
    enhanced_prompt = generate_fashion_prompt(prompt)
    print("\n--- Generated Photography Prompt ---\n")
    print(enhanced_prompt)

    # Route to the appropriate model
    if model == "finesse.fl.dev":
        image = run_flux_minimal(enhanced_prompt)
    elif model == "finesse.gi1":
        image = run_gpt(enhanced_prompt)
    else:
        raise ValueError(f"Unsupported model: {model}")

    return image


# Define the Gradio interface
interface = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Dropdown(
            choices=[("finesse.fl.dev", "finesse.fl.dev"), ("finesse.gi1", "finesse.gi1")],
            label="Model",
        ),
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Diffusion Model Image Generator",
    description="Enter a text prompt, choose a model, and click Generate to produce an image.",
    allow_flagging="never",
)

if __name__ == "__main__":
    interface.launch(share=True, auth=[("ramin", "ramin123"), ("peter", "peter123")])
