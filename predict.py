# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
import torch
sys.path.append("/style-aligned")
import sa_handler
from typing import List
from weights_downloader import WeightsDownloader
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import numpy as np
from diffusers.utils import load_image
import inversion
import tempfile
import shutil

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
        self.pipeline = pipeline

    def predict(
        self,
        prompt: str = Input(description="Input prompts, separated by newlines"),
        style_prompt: str = Input(description="Input style prompt. If using an image for style transfer, this should describe the style of that image"),
        negative_prompt: str = Input(description="Input negative prompt", default="low-resolution"),
        image: Path = Input(description="Input image for style transfer", default=None),
        image_subject: str = Input(description="Subject (not style) of image used for style transfer. Ignored if image not set.", default="None"),
        shared_score_shift: float = Input(description="Shared score shift (Actual value will take log of this). Higher value induces higher fidelity, set 1 for no shift", default=2, ge=0.01, le=10),
        shared_score_scale: float = Input(description="Shared score scale. Higher value induces higher, set 1 for no rescale", default=1.0, ge=0.0, le=1.0),
        guidance_scale: float = Input(description="Guidance scale", default=7.0, ge=1.1, le=20.0),
        width: int = Input(description="Output image width. Will be rounded to nearest 64", default=768, ge=256, le=2048),
        height: int = Input(description="Output image height. Will be rounded to nearest 64", default=768, ge=256, le=2048),
        num_inference_steps: int = Input(description="Number of inference steps", default=50, ge=10, le=150),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        if width%64 != 0:
            print('Width not divisible by 64, rounding to nearest 64')
            if width%64 < 32:
                width = width - width%64
            else:
                width = width + 64 - width%64

        if height%64 != 0:
            print('Height not divisible by 64, rounding to nearest 64')
            if height%64 < 32:
                height = height - height%64
            else:
                height = height + 64 - height%64
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # split prompt by newlines into an array of strings
        inputs = prompt.strip().splitlines()
        # if fewer negative prompts than expected, then repeat it
        neg_inputs = negative_prompt.strip().splitlines()
        if len(neg_inputs) < len(inputs):
            neg_inputs = [negative_prompt] * len(inputs)
        for i, input in enumerate(inputs):
            inputs[i] = f'{input}, {style_prompt}'
        
        if image is not None:
            tempdir = tempfile.mkdtemp()
            extension = os.path.splitext(image)[1]
            image_path = os.path.join(tempdir, f"image{extension}")

            shutil.copy(image, image_path)
            d_x = width//8
            d_y = height//8
            x0 = np.array(load_image(image_path).resize((width, height)))
            zts = inversion.ddim_inversion(self.pipeline, x0, style_prompt, num_inference_steps, 2)

            inputs.insert(0, f"{image_subject}, {style_prompt}")
            neg_inputs.insert(0, "")

            handler = sa_handler.Handler(self.pipeline)
            sa_args = sa_handler.StyleAlignedArgs(
                share_group_norm=True, share_layer_norm=True, share_attention=True,
                adain_queries=True, adain_keys=True, adain_values=False,
                shared_score_shift=np.log(shared_score_shift), shared_score_scale=shared_score_scale,)
            handler.register(sa_args)

            zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)

            g_cpu = torch.Generator(device='cpu')
            g_cpu.manual_seed(seed)
            latents = torch.randn(len(inputs), 4, d_y, d_x, device='cpu', generator=g_cpu,
                                dtype=self.pipeline.unet.dtype,).to('cuda:0')

            latents[0] = zT
            images = self.pipeline(
                inputs,
                latents=latents,
                callback_on_step_end=inversion_callback,
                guidance_scale=guidance_scale,
                negative_prompt=neg_inputs,
                # width=width,
                # height=height,
                num_inference_steps=num_inference_steps
            ).images
        else:
            handler = sa_handler.Handler(self.pipeline)
            sa_args = sa_handler.StyleAlignedArgs(
                share_group_norm=False,
                share_layer_norm=False,
                share_attention=True,
                adain_queries=True,
                adain_keys=True,
                adain_values=False
            )
            handler.register(sa_args)
            images = self.pipeline(
                prompt=inputs, 
                negative_prompt=neg_inputs,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images
            handler.remove()
            

        output_paths = []
        for i, im in enumerate(images):
            if image is not None and i == 0:
                # The first image is the image we passed in
                continue
            output_path = f"/tmp/output-{i}.png"
            im.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
