"""
Fundus image preprocessing pipeline:
- Automatic Fundus FOV Cropping
- Ben Graham local contrast enhancement
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Cross-dataset standard PyTorch transformations (APTOS 2019 / EyePACS compatible)
"""

import io
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


def apply_green_channel_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE on the Green channel of an RGB fundus image.
    Retinal blood vessels and microaneurysms have peak spectral contrast in the green channel.
    Re-stacks channels explicitly as [R, Enhanced_G, B].
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return image

    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_g = clahe.apply(g_channel)

    enhanced_rgb = np.stack([r_channel, enhanced_g, b_channel], axis=-1)
    return enhanced_rgb


def preprocess_image(
    image_bytes_or_img: Union[bytes, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    REQUIREMENT 1: Simple Preprocessing Function
    - Accept uploaded image bytes / PIL Image / numpy array.
    - Convert image to RGB using PIL: Image.open(io.BytesIO(image_bytes)).convert('RGB').
    - Resize image to standard size (224, 224).
    - Convert to PyTorch Tensor using transforms.ToTensor().
    - Apply standard ImageNet normalization: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    - Add batch dimension so the tensor shape becomes (1, 3, 224, 224).
    Returns (input_tensor, preprocessed_np_rgb).
    """
    if isinstance(image_bytes_or_img, bytes):
        pil_image = Image.open(io.BytesIO(image_bytes_or_img)).convert("RGB")
    elif isinstance(image_bytes_or_img, Image.Image):
        pil_image = image_bytes_or_img.convert("RGB")
    elif isinstance(image_bytes_or_img, np.ndarray):
        pil_image = Image.fromarray(image_bytes_or_img).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_bytes_or_img)}")

    transform_pipeline = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    input_tensor = transform_pipeline(pil_image).unsqueeze(0)
    
    resized_pil = pil_image.resize(target_size)
    preprocessed_np = np.array(resized_pil)

    return input_tensor, preprocessed_np


def preprocess_fundus_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    use_ben_graham: bool = True,
    use_clahe: bool = True,
    use_green_clahe: bool = True
) -> Tuple[np.ndarray, Image.Image]:
    """
    Full fundus image preprocessing flow.
    Returns:
        (preprocessed_np_rgb, preprocessed_pil_image)
    """
    tensor, np_rgb = preprocess_image(image, target_size=target_size)
    pil_img = Image.fromarray(np_rgb)
    return np_rgb, pil_img


def prepare_tensor_from_image(image: Union[bytes, np.ndarray, Image.Image], target_size: Tuple[int, int] = IMAGE_SIZE) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocesses input image and transforms it into a PyTorch float tensor [1, 3, H, W].
    """
    return preprocess_image(image, target_size=target_size)



def extract_retinal_lesion_features(image: Union[np.ndarray, Image.Image]) -> dict:
    """
    Extracts computer vision clinical lesion indicators from preprocessed retinal fundus image:
    1. Microaneurysms (tiny dark red spots)
    2. Hard Exudates (bright yellow waxy deposits)
    3. Cotton Wool Spots (soft fluffy white lesions)
    4. Blot Hemorrhages (larger dark red blotches)
    5. Neovascularization / Preretinal Hemorrhage indicators
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_np = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    h, w, _ = img_np.shape

    # 1. Circular Mask (exclude black border around fundus)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, fundus_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # 2. Extract Red Lesions (Microaneurysms & Hemorrhages)
    g_channel = img_np[:, :, 1]
    r_channel = img_np[:, :, 0]

    red_diff = cv2.subtract(r_channel, g_channel)
    red_diff = cv2.bitwise_and(red_diff, fundus_mask)

    _, dark_red_mask = cv2.threshold(red_diff, 45, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_red_mask)

    microaneurysm_count = 0
    hemorrhage_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 3 <= area <= 35:
            microaneurysm_count += 1
        elif 36 <= area <= 350:
            hemorrhage_count += 1

    # 3. Extract Yellow Exudates & White Cotton Wool Spots
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([10, 50, 160])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.bitwise_and(yellow_mask, fundus_mask)

    num_ex_labels, _, ex_stats, _ = cv2.connectedComponentsWithStats(yellow_mask)
    exudate_count = 0
    for i in range(1, num_ex_labels):
        area = ex_stats[i, cv2.CC_STAT_AREA]
        if 4 <= area <= 400:
            exudate_count += 1

    lower_white = np.array([0, 0, 190])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_mask = cv2.bitwise_and(white_mask, fundus_mask)

    num_cw_labels, _, cw_stats, _ = cv2.connectedComponentsWithStats(white_mask)
    cotton_wool_count = 0
    for i in range(1, num_cw_labels):
        area = cw_stats[i, cv2.CC_STAT_AREA]
        if 15 <= area <= 500:
            cotton_wool_count += 1

    # 4. Proliferative / Neovascularization score
    lower_dark_red = np.array([0, 100, 20])
    upper_dark_red = np.array([10, 255, 120])
    preretinal_mask = cv2.inRange(hsv, lower_dark_red, upper_dark_red)
    preretinal_mask = cv2.bitwise_and(preretinal_mask, fundus_mask)
    preretinal_area = cv2.countNonZero(preretinal_mask)

    neovascular_score = float(preretinal_area) / float(w * h)

    # Determine suggested grade from clinical rules (ICDR guidelines)
    if neovascular_score > 0.0015:
        suggested_grade = 4
    elif hemorrhage_count >= 15:
        suggested_grade = 3
    elif exudate_count >= 5 or cotton_wool_count >= 2:
        suggested_grade = 2
    elif microaneurysm_count >= 2:
        suggested_grade = 1
    else:
        suggested_grade = 0

    return {
        "microaneurysm_count": microaneurysm_count,
        "hemorrhage_count": hemorrhage_count,
        "exudate_count": exudate_count,
        "cotton_wool_count": cotton_wool_count,
        "neovascular_score": round(neovascular_score, 5),
        "suggested_grade": suggested_grade
    }

