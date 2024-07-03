#!/usr/bin/env python
#patch 0.04
#Func()   
#Pruned Dalle4K Space 
import os
import random
import uuid
import json
import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, KDPM2AncestralDiscreteScheduler, AutoencoderKL
from typing import Tuple

# Check for the Model Base..

bad_words = json.loads(os.getenv('BAD_WORDS', "[]"))
bad_words_negative = json.loads(os.getenv('BAD_WORDS_NEGATIVE', "[]"))
default_negative = os.getenv("default_negative", "")

def check_text(prompt, negative=""):
    for i in bad_words:
        if i in prompt:
            return True
    for i in bad_words_negative:
        if i in negative:
            return True
    return False


DESCRIPTIONx = """

## GALLO 3XL 🏜️

Caution: This space consumes a high amount of GPU resources and may lead to the depletion of your GPU quota.
"""
style_list = [
    {
        "name": "3840 x 2160",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },  
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt}. emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt}. anime style, key visual, vibrant, studio anime, highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt}. octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "3840 x 2160"

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

DESCRIPTION = """"""
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>⚠️Running on CPU, This may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_IMAGES_PER_PROMPT = 1

if torch.cuda.is_available():
    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe2 = StableDiffusionXLPipeline.from_pretrained(
        "Corcelio/mobius",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe3 = StableDiffusionXLPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        pipe2.enable_model_cpu_offload()
        pipe3.enable_model_cpu_offload()
    else:
        pipe.to(device)    
        pipe2.to(device)
        pipe3.to(device)
        print("Loaded on Device!")
    
    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe2.unet = torch.compile(pipe2.unet, mode="reduce-overhead", fullgraph=True)
        pipe3.unet = torch.compile(pipe3.unet, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    style: str = DEFAULT_STYLE_NAME,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    if check_text(prompt, negative_prompt):
        raise ValueError("Prompt contains restricted words.")
    
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    if not use_negative_prompt:
        negative_prompt = ""  # type: ignore
    negative_prompt += default_negative    

    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": 23,
        "generator": generator,
        "num_images_per_prompt": 1,  # Set to generate 1 image per model
        "use_resolution_binning": use_resolution_binning,
        "output_type": "pil",
    }
    
    # Generate one image from each model
    image1 = pipe(**options).images[0]
    image2 = pipe2(**options).images[0]
    image3 = pipe3(**options).images[0]

    images = [image1, image2, image3]

    image_paths = [save_image(img) for img in images]
    return image_paths, seed

examples = [
    "A blurry photo of mustang, with orange and yellow colors and a blurred background. Silhouettes of mountains in the foreground against a beautiful sunset. High resolution, captured with a Canon EOS R5 at F8 and ISO2. No people are depicted. --ar 128:85 --v 6.0 --style raw",
    "Child in the style of bold character designs, interactive, hannah yata, light sky and pink, bold colorism, storybook-like, animated gifs ",
    "Food photography of a milk shake with flying strawberrys against a pink background, professionally studio shot with cinematic lighting. The image is in the style of a professional studio shot --ar 85:128 --v 6.0 --style raw",
    "Masterpiece, best quality, 1girl, solo, (loli, child), blue hair, ponytail hair, [pink|blue] eyes, white theme, blue theme, white T-shirt, shorts, medium breast, navel, (sky, sunlight, ocean, from below:1.36), standing, cowboy shot, <lora:add_detail:1>, 8k, UHD, HDR,(Masterpiece:1.5), (best quality:1.5)"
]

css = '''
.gradio-container{max-width: 600px !important}
h1{text-align:center}
'''
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown(DESCRIPTIONx)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run")
        result = gr.Gallery(label="Result", columns=1, preview=True)
    with gr.Accordion("Advanced Options", open=False, visible=False ):
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use Negative Prompt", value=True)
            negative_prompt = gr.Text(
                label="Negative Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter a negative prompt",
                value="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                visible=True,
            )
        with gr.Row():
            num_inference_steps = gr.Slider(
                label="Steps",
                minimum=10,
                maximum=60,
                step=1,
                value=23,
            )
        with gr.Row():
            num_images_per_prompt = gr.Slider(
                label="Images",
                minimum=1,
                maximum=5,
                step=1,
                value=3,  # Set default to 3 images
                visible=False,  # Hide this slider since we are fixing the value to 3 images
            )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
            visible=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=20.0,
                step=0.1,
                value=6,
            )
    with gr.Row(visible=False):
        style_selection = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=STYLE_NAMES,
            value=DEFAULT_STYLE_NAME,
            label="Image Style",
        )
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            style_selection,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )


    gr.Markdown("⚠️ users are accountable for the content they generate and are responsible for ensuring it meets appropriate ethical standards.")

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
