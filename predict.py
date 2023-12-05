# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
import torch
sys.path.append("style-aligned")
import sa_handler
from typing import List
from weights_downloader import WeightsDownloader
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

SDXL_CACHE = "sdxl-cache"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        WeightsDownloader.download_if_not_exists(SDXL_URL, SDXL_CACHE)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            SDXL_CACHE,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            scheduler=scheduler
        ).to("cuda")
        pipeline.enable_xformers_memory_efficient_attention()
        handler = sa_handler.Handler(pipeline)
        sa_args = sa_handler.StyleAlignedArgs(
            share_group_norm=False,
            share_layer_norm=False,
            share_attention=True,
            adain_queries=True,
            adain_keys=True,
            adain_values=False
        )
        handler.register(sa_args)
        self.pipeline = pipeline

    def predict(
        self,
        prompt: str = Input(description="Input prompts, separated by newlines"),
        negative_prompt: str = Input(description="Input negative prompt", default="low-resolution"),
        width: int = Input(description="Output image width", default=768, ge=256, le=2048),
        height: int = Input(description="Output image height", default=768, ge=256, le=2048),
        num_inference_steps: int = Input(description="Number of inference steps", default=50, ge=1, le=150),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # slit prompt by newlines into an array of strings
        inputs = prompt.strip().splitlines()
        # if fewer negative promtps than expected, then repeat it
        neg_inputs = negative_prompt.strip().splitlines()
        if len(neg_inputs) < len(inputs):
            neg_inputs = [negative_prompt] * len(inputs)
        images = self.pipeline(
            prompt=inputs, 
            negative_prompt=neg_inputs,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/output-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
