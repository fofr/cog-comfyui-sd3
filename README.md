# Cog template for Stable Diffusion 3 (ComfyUI implementation)

[![Replicate demo and cloud API](https://replicate.com/stability-ai/stable-diffusion-3/badge)](https://replicate.com/stability-ai/stable-diffusion-3)

> Run Stable Diffusion 3 with an API on Replicate:
>
> https://replicate.com/stability-ai/stable-diffusion-3

This is an implementation of Stability AI's [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3-medium) as a [Cog](https://github.com/replicate/cog) model.

Stable Diffusion 3 has [a non-commercial license](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE). [stability-ai/stable-diffusion-3](https://replicate.com/stability-ai/stable-diffusion-3) on Replicate is licensed and can be used for commercial work, but if you want to use custom versions built from this repository for commercial work, you'll need to buy [a commercial license](https://stability.ai/license).

This Cog implementation uses ComfyUI and is based on https://github.com/fofr/cog-comfyui. If you prefer to use diffusers, check out our diffusers implementation: https://github.com/replicate/cog-stable-diffusion-3

## Local Usage

For a local prediction, run:

```bash
cog predict -i prompt="a photo of a cool dog"
```


## Accessing the local ComfyUI server

The local ComfyUI server is available at [http://localhost:8000](http://localhost:8000).

1. **GPU Machine**: Start the Cog container and expose port 8188 so you can access it:
```sh
sudo cog run -p 8188 bash
```

2. **Inside Cog Container**: Now that we have access to the Cog container, we start the server, binding to all network interfaces:
```sh
cd ComfyUI/
python main.py --listen 0.0.0.0
```

3. **Local Machine**: Access the server using your GPU machine's IP and the exposed port (8188):
`http://<gpu-ip>:8188`

Now if you visit `http://<gpu-ip>:8188` you’ll see the classic ComfyUI server. You can load in the `workflow_api.json` or `workflow_ui.json` file to get started.


You’ll also need a copy of the Stable Diffusion 3 weights.

First get access by filling out the form on the [weights page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main). You’ll need to download `sd3_medium_incl_clips_t5xxlfp8.safetensors` to the `ComfyUI/models/checkpoints` folder.
