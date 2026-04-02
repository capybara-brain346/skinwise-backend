from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image).astype(np.float32)

    img_array = img_array / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    img_array = np.transpose(img_array, (2, 0, 1))

    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    return img_array