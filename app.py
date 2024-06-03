#!/usr/bin/env python

import os
import random
import uuid

import gradio as gr
import numpy as np
from PIL import Image
import spaces
from typing import Tuple
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

DESCRIPTION = """ """

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

MAX_SEED = np.iinfo(np.int32).max

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU, This may not work on CPU.</p>"

USE_TORCH_COMPILE = 0
ENABLE_CPU_OFFLOAD = 0

style_list = [

    {
    "name": "3840 x 2160",
    "prompt": "hyper-realistic 8K image of {prompt} . ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
    "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
    "name": "2560 × 1440",
    "prompt": "hyper-realistic 4K image of {prompt} . ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
    "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
    "name": "HDR",
    "prompt": "HDR photo of {prompt} . high dynamic range, vivid colors, sharp contrast, realistic, detailed, high resolution, professional",
    "negative_prompt": "dull, low contrast, blurry, unrealistic, cartoonish, ugly, deformed",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photo",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
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

@spaces.GPU(enable_queue=True)
def stab(
    prompt: str,
    negative_prompt: str = "",
    style: str = DEFAULT_STYLE_NAME,
    use_negative_prompt: bool = False,
    num_inference_steps: int = 30,
    num_images_per_prompt: int = 2,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))

    if not use_negative_prompt:
        negative_prompt = ""
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)

    prior.enable_model_cpu_offload()
    prior_output = prior(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps
    )

    decoder.enable_model_cpu_offload()
    images = decoder(
        image_embeddings=prior_output.image_embeddings.to(torch.float16),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=10
    ).images

    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths, seed

examples = [
		
	"3d image, cute girl, in the style of Pixar --ar 1:2 --stylize 750, 4K resolution highlights, Sharp focus, octane render, ray tracing, Ultra-High-Definition, 8k, UHD, HDR, (Masterpiece:1.5), (best quality:1.5)",
    "(Pirate ship sailing into a bioluminescence sea with a galaxy in the sky), epic, 4k, ultra, the space scene with planets and stars, in the style of ethereal escapism, richly colored skies, vibrant worlds --ar 8:5",
    "Thin burger, realistic photo (without tomato or any other ingredient), smoky flavor, 4K resolution highlights every texture, providing an incredible and appetizing visual experience",
    "A galaxy with blue water, a red star and many planets in one view, in the style of digital fantasy nebulae and cosmos, light black and violet, realistic nebulae paintings, james paick, steve henderson, ue5, cosmic horror --ar 8:5",
    "A dark night sky with thick, dense clouds and stars in the background. The main focus is on one of these large cloud formations that has been stylized to resemble an ancient dragon. There's no moon or other celestial bodies visible in the sky. This scene conveys mystery and magic, with the dark blue glow from distant galaxies adding depth and contrast to the night landscape. --ar 8:5 --v 5.2 --style raw"

]

css = '''
.gradio-container{max-width: 560px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''

with gr.Blocks(css=css, theme="xiaobaiyuan/theme_brief") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=False,
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
    with gr.Accordion("Advanced options", open=False):
        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True, visible=True)
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
            value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
            visible=True,
        )
        with gr.Row():
            num_inference_steps = gr.Slider(
                label="Steps",
                minimum=10,
                maximum=60,
                step=1,
                value=30,
            )
        with gr.Row():
            num_images_per_prompt = gr.Slider(
                label="Images",
                minimum=1,
                maximum=5,
                step=1,
                value=2,
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
    with gr.Row(visible=True):
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
        fn=stab,
        cache_examples=False,
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
        fn=stab,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            num_inference_steps,
            num_images_per_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(show_api=False, debug=False, share=True)
