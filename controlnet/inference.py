from PIL import Image

def generate_image(pipe, prompt, negative_prompt, control_image):

    if not isinstance(control_image, Image.Image):
        control_image = Image.fromarray(control_image)

    
    prompt = """
        A clean electrical circuit schematic.
        Technical engineering drawing.
        Preserve every wire and every electronic component.
        Use standard electrical symbols.
        Black lines on a pure white background.
        Flat 2D schematic.
        No perspective.
        No shading.
        No textures.
        No artistic style.
        No extra components.
        """

    negative_prompt = """
        photo,
        realistic,
        painting,
        3d,
        shadow,
        texture,
        noise,
        extra wires,
        extra components,
        extra circles,
        distorted,
        blurry,
        low quality
        """

    result = pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        image = control_image, 
        num_inference_steps = 20,
        guidance_scale = 7.5
    )

    return result.images[0]