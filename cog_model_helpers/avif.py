from PIL import Image
import os


def handle_avif_inputs(path_to_image):
    import pillow_avif
    print(f"Converting AVIF to PNG: {path_to_image}")
    with Image.open(path_to_image) as img:
        png_path = os.path.splitext(path_to_image)[0] + ".png"
        img.save(png_path, "PNG")

    return png_path, ".png"
