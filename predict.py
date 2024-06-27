# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from weights_downloader import WeightsDownloader

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]
mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow_api.json"

aspect_ratios = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        weights_downloader = WeightsDownloader()
        weights_downloader.download(
            "sd3_medium_incl_clips_t5xxlfp8.safetensors",
            "https://weights.replicate.delivery/default/comfy-ui/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors.tar",
            "ComfyUI/models/checkpoints",
        )

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(self, workflow, **kwargs):
        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["71"]["inputs"]
        negative_prompt["text"] = kwargs["negative_prompt"]

        sampler = workflow["271"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["steps"] = kwargs["steps"]
        sampler["cfg"] = kwargs["cfg"]

        if kwargs["image_filename"]:
            load_image = workflow["275"]
            load_image["inputs"]["image"] = kwargs["image_filename"]
            sampler["denoise"] = kwargs["prompt_strength"]

            with Image.open(os.path.join(INPUT_DIR, kwargs["image_filename"])) as img:
                original_width, original_height = img.size

            input_aspect_ratio = original_width / original_height
            closest_ratio = min(
                aspect_ratios.items(),
                key=lambda x: abs(x[1][0] / x[1][1] - input_aspect_ratio),
            )

            kwargs["width"], kwargs["height"] = closest_ratio[1]
            workflow["277"]["inputs"]["width"] = kwargs["width"]
            workflow["277"]["inputs"]["height"] = kwargs["height"]
        else:
            sampler["denoise"] = 1
            sampler["latent_image"] = ["135", 0]
            empty_latent_image = workflow["135"]["inputs"]
            empty_latent_image["width"] = kwargs["width"]
            empty_latent_image["height"] = kwargs["height"]
            del workflow["275"]  # remove the load image node
            del workflow["277"]  # remove the image resize node
            del workflow["278"]  # remove the VAE encode node

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        aspect_ratio: str = Input(
            choices=list(aspect_ratios.keys()),
            default="1:1",
            description="The aspect ratio of your output image. This value is ignored if you are using an input image.",
        ),
        cfg: float = Input(
            description="The guidance scale tells the model how similar the output should be to the prompt.",
            le=20,
            ge=0,
            default=3.5,
        ),
        image: Path = Input(
            description="Input image for image to image mode. The aspect ratio of your output will match this image.",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength (or denoising strength) when using image to image. 1.0 corresponds to full destruction of information in image.",
            ge=0.0,
            le=1.0,
            default=0.85,
        ),
        steps: int = Input(
            description="Number of steps to run the sampler for.",
            ge=1,
            le=28,
            default=28,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
        negative_prompt: str = Input(
            description="Negative prompts do not really work in SD3. Using a negative prompt will change your output in unpredictable ways.",
            default="",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        width, height = aspect_ratios.get(aspect_ratio)

        if image:
            file_extension = os.path.splitext(image)[1].lower()
            image_filename = f"input{file_extension}"
            self.handle_input_file(image, image_filename)
        else:
            image_filename = None

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            cfg=cfg,
            width=width,
            height=height,
            prompt_strength=prompt_strength,
            image_filename=image_filename,
            steps=steps,
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(workflow)

        files = self.comfyUI.get_files(OUTPUT_DIR)
        if len(files) == 0:
            raise Exception("No output")

        return optimise_images.optimise_image_files(
            output_format, output_quality, files
        )
