import torch
from diffusers import ZImagePipeline

from datetime import datetime

def filename_timestamp():
    """
    Returns a filesystem-safe timestamp string, e.g.:
    2025-03-03_14-22-59
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "./Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

prompt = "a high quality photo of a golden retriever playing in the snow, in the woods"
prompts = [
    "a man firing pineapples from a gun in a surreal action scene",
    "a man wearing a pineapple-themed outfit while shooting pineapples from a weapon",
    "a dynamic photo of a man launching pineapples from a gun in a humorous, surreal setting",
    "a cinematic shot of a man dressed in fruit-themed clothing firing pineapples like projectiles",
    "a high-quality image of a man shooting pineapple-shaped ammo from a stylized gun"
]


i = 1
for prompt in prompts:
    # 2. Generate Image
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    now = filename_timestamp()
    image.save(f"{i}_image_{now}.png")
    i = i + 1
