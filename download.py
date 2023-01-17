# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

import os

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
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
        use_auth_token=HF_AUTH_TOKEN
    )

if __name__ == "__main__":
    download_model()
