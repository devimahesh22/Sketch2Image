from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetPipeline
import torch

def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float32
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32
    )

    return pipe