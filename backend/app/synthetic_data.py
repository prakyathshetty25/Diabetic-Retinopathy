"""
Synthetic Fundus Image Generator for Universal Retinal Screening framework.
Generates realistic eye fundus scans across all 5 DR severity grades (0: No DR to 4: Proliferative DR)
with synthetic optic disc, vascular tree, microaneurysms, hard exudates, and hemorrhages.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from typing import Dict, Tuple


def create_synthetic_fundus(grade: int = 0, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Generates a realistic synthetic fundus photograph corresponding to DR severity grade 0-4.
    """
    h, w = size
    # 1. Base orange-red retinal background with dark radial vignette
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    
    center_x, center_y = w // 2, h // 2
    radius = int(min(h, w) * 0.46)

    # Circular mask
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    fundus_mask = dist_from_center <= radius

    # Retinal background color: Deep orange/red RGB ~ (195, 75, 25)
    for y in range(h):
        for x in range(w):
            if fundus_mask[y, x]:
                # Radial distance factor for realistic peripheral darkening
                r_factor = 1.0 - 0.4 * (dist_from_center[y, x] / radius)**2
                red = int(210 * r_factor)
                green = int(85 * r_factor)
                blue = int(25 * r_factor)
                bg[y, x] = [red, green, blue]

    # Convert to PIL for drawing vascular structures and lesions
    pil_img = Image.fromarray(bg)
    draw = ImageDraw.Draw(pil_img)

    # 2. Optic Disc (bright yellow-white oval in nasal quadrant)
    disc_x, disc_y = int(center_x - radius * 0.45), int(center_y - radius * 0.05)
    disc_r = int(radius * 0.14)
    draw.ellipse(
        [disc_x - disc_r, disc_y - disc_r, disc_x + disc_r, disc_y + disc_r],
        fill=(255, 240, 190),
        outline=(255, 220, 150)
    )

    # 3. Retinal Vascular Tree (Dark red branching arcade from optic disc)
    vessel_color = (130, 20, 10)
    
    # Superior & Inferior vascular arcades
    branches = [
        [(disc_x, disc_y), (disc_x + 50, disc_y - 80), (disc_x + 150, disc_y - 120), (disc_x + 220, disc_y - 80)],
        [(disc_x, disc_y), (disc_x + 60, disc_y + 80), (disc_x + 160, disc_y + 110), (disc_x + 230, disc_y + 70)],
        [(disc_x, disc_y), (disc_x - 30, disc_y - 70), (disc_x - 60, disc_y - 120)],
        [(disc_x, disc_y), (disc_x - 30, disc_y + 70), (disc_x - 60, disc_y + 110)]
    ]
    
    for branch in branches:
        for i in range(len(branch) - 1):
            width = max(2, 6 - i * 2)
            draw.line([branch[i], branch[i+1]], fill=vessel_color, width=width)

    # 4. Macula (Darker foveal avascular zone in temporal region)
    macula_x, macula_y = int(center_x + radius * 0.15), int(center_y)
    macula_r = int(radius * 0.12)
    draw.ellipse(
        [macula_x - macula_r, macula_y - macula_r, macula_x + macula_r, macula_y + macula_r],
        fill=(140, 45, 15)
    )

    # 5. Add Grade-Specific Lesions
    np_fundus = np.array(pil_img)

    if grade >= 1:
        # Mild: Add Microaneurysms (tiny dark red dots)
        num_ma = 8 if grade == 1 else (20 if grade == 2 else 45)
        for _ in range(num_ma):
            rx = int(center_x + np.random.uniform(-0.6, 0.6) * radius)
            ry = int(center_y + np.random.uniform(-0.6, 0.6) * radius)
            if fundus_mask[min(h-1, max(0, ry)), min(w-1, max(0, rx))]:
                cv2.circle(np_fundus, (rx, ry), np.random.randint(2, 4), (110, 10, 5), -1)

    if grade >= 2:
        # Moderate: Add Hard Exudates (bright yellow waxy deposits) & Cotton Wool Spots
        num_exudates = 15 if grade == 2 else (35 if grade == 3 else 60)
        for _ in range(num_exudates):
            rx = int(macula_x + np.random.uniform(-0.4, 0.4) * radius)
            ry = int(macula_y + np.random.uniform(-0.4, 0.4) * radius)
            if fundus_mask[min(h-1, max(0, ry)), min(w-1, max(0, rx))]:
                cv2.circle(np_fundus, (rx, ry), np.random.randint(3, 7), (250, 245, 140), -1)

        # Cotton Wool Spots (soft white fluffy lesions)
        for _ in range(5):
            rx = int(center_x + np.random.uniform(-0.5, 0.5) * radius)
            ry = int(center_y + np.random.uniform(-0.5, 0.5) * radius)
            cv2.ellipse(np_fundus, (rx, ry), (12, 7), np.random.randint(0, 180), 0, 360, (230, 230, 210), -1)

    if grade >= 3:
        # Severe: Add Blot Hemorrhages (> 20 in 4 quadrants) and Venous Beading
        for _ in range(25):
            rx = int(center_x + np.random.uniform(-0.7, 0.7) * radius)
            ry = int(center_y + np.random.uniform(-0.7, 0.7) * radius)
            if fundus_mask[min(h-1, max(0, ry)), min(w-1, max(0, rx))]:
                cv2.ellipse(np_fundus, (rx, ry), (np.random.randint(6, 12), np.random.randint(4, 9)),
                            np.random.randint(0, 180), 0, 360, (100, 5, 5), -1)

    if grade == 4:
        # Proliferative: Add Neovascularization (fragile new vessel networks) & Preretinal Hemorrhage
        for _ in range(12):
            rx = int(disc_x + np.random.uniform(-0.3, 0.5) * radius)
            ry = int(disc_y + np.random.uniform(-0.3, 0.5) * radius)
            pts = np.array([
                [rx, ry],
                [rx + np.random.randint(-20, 20), ry + np.random.randint(-20, 20)],
                [rx + np.random.randint(-30, 30), ry + np.random.randint(-30, 30)]
            ], np.int32)
            cv2.polylines(np_fundus, [pts], False, (160, 15, 10), 2)

        # Vitreous/Preretinal boat-shaped hemorrhage
        cv2.ellipse(np_fundus, (center_x + 40, center_y + 30), (35, 18), 15, 0, 180, (90, 0, 0), -1)

    # Slight blur to blend synthetic elements smoothly into medical image aesthetic
    blurred = cv2.GaussianBlur(np_fundus, (3, 3), 0)
    return Image.fromarray(blurred)


def image_to_base64(img: Image.Image) -> str:
    """Converts PIL image to base64 data string."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"
