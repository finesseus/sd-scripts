import gradio as gr
from PIL import Image

from flux_minimal_inference import flux_minimal
from gpt_desc import generate_fashion_prompt, generate_image_2, generate_image_from_prompt


# Import or define your model functions
# from diffusion_model import flux_minimal, generate_image_from_prompt, generate_fashion_prompt


def run_flux_minimal(prompt: str) -> Image.Image:
    return flux_minimal(prompt)


def run_gpt(prompt: str) -> Image.Image:
    return generate_image_2(prompt)


def generate_and_update(prompt: str, model: str, history):
    """
    Generate an image, update the history state, and return outputs for image, history state, and gallery data.
    History is a list of tuples: (PIL.Image, caption).
    """
    # Enhance the prompt
    enhanced_prompt = generate_fashion_prompt(prompt, model)
    # Route to the appropriate model
    if model == "finesse.fl.dev":
        image = run_flux_minimal(enhanced_prompt)
    elif model == "finesse.gi1":
        image = run_gpt(enhanced_prompt)
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Build caption
    caption = f"Model: {model.split('.', 1)[1]} | Prompt: {prompt}"
    # Update history list
    new_history = history + [(image, caption)]

    # Return image, updated state, and gallery data
    return image, new_history, new_history


# Build the Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Finesse Design Demo
        Enter a text prompt, choose a model, and click **Generate** to produce an image.
        Past generations will appear below in the history gallery.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt")
            model_input = gr.Dropdown(
                choices=["finesse.fl.dev", "finesse.gi1"],
                value="finesse.fl.dev",
                label="Model",
            )
            generate_btn = gr.Button("Generate")
        with gr.Column(scale=4):
            output_image = gr.Image(type="pil", label="Generated Image")

    # State to keep track of history
    history_state = gr.State([])
    # Gallery to display history with two columns
    history_gallery = gr.Gallery(label="Generation History", columns=2, height="auto")

    # Wire up the button: update image and history state only
    generate_btn.click(
        fn=generate_and_update, inputs=[prompt_input, model_input, history_state], outputs=[output_image, history_state]
    )

    # Separate listener to update gallery when history_state changes
    def update_gallery(history):
        return history

    history_state.change(fn=update_gallery, inputs=[history_state], outputs=[history_gallery])

if __name__ == "__main__":
    # demo.launch(share=True)
    demo.launch(share=True, auth=[("ramin", "ramin123"), ("peter", "peter123")])
