import torch
from torch import autocast
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import base64
from io import BytesIO
import os
import PIL

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = "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"
    
    dpmscheduler = DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        trained_betas=None,
        thresholding=False,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,
    )
    model = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        scheduler=dpmscheduler,
        torch_dtype=torch.float16,
        use_auth_token=HF_AUTH_TOKEN
    ).to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    init_image_base64 = model_inputs.get('init_image_base64', None)
    mask_image_base64 = model_inputs.get('mask_image_base64', None)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    num_inference_steps = model_inputs.get('steps', 50)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    input_seed = model_inputs.get("seed",None)


    init_image_encoded = init_image_base64.encode('utf-8')
    init_image_bytes = BytesIO(base64.b64decode(init_image_encoded))
    init_image = PIL.Image.open(init_image_bytes)
    
    mask_image_encoded = mask_image_base64.encode('utf-8')
    mask_image_bytes = BytesIO(base64.b64decode(mask_image_encoded))
    mask_image = PIL.Image.open(mask_image_bytes)


    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    with autocast("cuda"):
        image = model(prompt,image=init_image,mask_image=mask_image,height=height,width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator).images[0]
    
    buffered = BytesIO()
    image.save(buffered,format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
