"""
Fundus image preprocessing pipeline:
- Automatic Fundus FOV Cropping
- Ben Graham local contrast enhancement
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Cross-dataset standard PyTorch transformations (APTOS 2019 / EyePACS compatible)
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Tuple, Union
import logging

from app.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


def crop_fundus_fov(image: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crops dark black border surrounding fundus retinal scans (Ben Graham method).
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image provided for FOV cropping.")

    if image.ndim == 2:
        mask = image > tol
        if not np.any(mask):
            return image
        return image[np.ix_(mask.any(1), mask.any(0))]
    
    # Color image (H, W, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    
    check_dims = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape
    if check_dims[0] == 0 or check_dims[1] == 0:
        return image  # Fallback if image is too dark or invalid
    
    img1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    
    return np.stack([img1, img2, img3], axis=-1)


def apply_ben_graham_enhancement(image: np.ndarray, sigma: int = 30) -> np.ndarray:
    """
    Enhances microaneurysms and lesion contrast using Ben Graham's method:
    Rescaled = 4 * Image - 4 * GaussianBlur(Image, sigma) + 128
    """
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    enhanced = cv2.addWeighted(image, 4, blur, -4, 128)
    return enhanced


def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on LAB color space L-channel.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def preprocess_fundus_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    use_ben_graham: bool = True,
    use_clahe: bool = True
) -> Tuple[np.ndarray, Image.Image]:
    """
    Full fundus image preprocessing flow.
    Returns:
        (preprocessed_np_rgb, preprocessed_pil_image)
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            img_np = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            img_np = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # 1. FOV Cropping
    try:
        cropped = crop_fundus_fov(img_np)
    except Exception as e:
        logger.warning(f"FOV cropping failed, using original: {e}")
        cropped = img_np

    # 2. Resizing to standard dimensions
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    # 3. CLAHE enhancement
    if use_clahe:
        enhanced = apply_clahe(resized)
    else:
        enhanced = resized

    # 4. Optional Ben Graham enhancement blend for deep feature extraction
    if use_ben_graham:
        bg_enhanced = apply_ben_graham_enhancement(enhanced)
        final_rgb = cv2.addWeighted(enhanced, 0.7, bg_enhanced, 0.3, 0)
    else:
        final_rgb = enhanced

    pil_img = Image.fromarray(final_rgb)
    return final_rgb, pil_img


def get_inference_transforms():
    """
    Standard PyTorch normalization transform for preprocessed fundus images.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def prepare_tensor_from_image(image: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocesses input image and transforms it into a PyTorch float tensor [1, 3, H, W].
    """
    preprocessed_np, pil_img = preprocess_fundus_image(image)
    transforms = get_inference_transforms()
    tensor = transforms(pil_img).unsqueeze(0)  # Shape: [1, 3, 512, 512]
    return tensor, preprocessed_np
