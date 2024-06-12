# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
from PIL import Image, ExifTags
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
        check_orientation: bool = True,
    ):
        image = Image.open(input_file)

        if check_orientation:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = dict(image._getexif().items())

                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (KeyError, AttributeError):
                # EXIF data does not have orientation
                # Do not rotate
                pass

        image.save(os.path.join(INPUT_DIR, filename))

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
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
        return aspect_ratios.get(aspect_ratio)

    def update_workflow(self, workflow, **kwargs):
        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]
        negative_prompt = workflow["71"]["inputs"]
        negative_prompt["text"] = kwargs["negative_prompt"]
        sampler = workflow["271"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["cfg"]
        empty_latent_image = workflow["135"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        aspect_ratio: str = Input(
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1",
        ),
        cfg: float = Input(
            description="The guidance scale tells the model how similar the output should be to the prompt.",
            le=20,
            ge=0,
            default=4.5,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            cfg=cfg,
            width=width,
            height=height,
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
