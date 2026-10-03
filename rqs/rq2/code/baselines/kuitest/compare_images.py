from PIL import Image

def compare_images(image_path1: str, image_path2: str) -> bool:
    with Image.open(image_path1) as img1, Image.open(image_path2) as img2:
        if img1.size != img2.size or img1.mode != img2.mode:
            return False
        return img1.tobytes() == img2.tobytes()
