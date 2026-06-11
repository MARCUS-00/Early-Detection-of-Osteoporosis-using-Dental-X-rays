'''
Classical-CV pseudo-ground-truth mask generation — ported from Unet_Extraction.ipynb.
IMPORTANT: These masks are PSEUDO-GROUND-TRUTH (Sobel + morphology), NOT human annotations.
Used to train the U-Net in lieu of manually labelled data.
'''
from pathlib import Path
import cv2
import numpy as np


def directional_gradient(img: np.ndarray) -> np.ndarray:
    '''Multi-directional gradient magnitude, normalised to uint8.
    Sobel x/y + diagonal approximations -> max of all four magnitudes.
    Source: Unet_Extraction.ipynb — ported verbatim.'''
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    diag1  = (sobelx + sobely) / 2
    diag2  = (sobelx - sobely) / 2
    G = np.maximum(np.maximum(np.abs(sobelx), np.abs(sobely)),
                   np.maximum(np.abs(diag1),  np.abs(diag2)))
    return cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def lower_boundary_mask(img_gray: np.ndarray) -> np.ndarray:
    '''Pixels >= 90th-percentile gradient -> 255; apply MORPH_OPEN 3x3.
    Source: Unet_Extraction.ipynb — ported verbatim.'''
    G = directional_gradient(img_gray)
    threshold = np.percentile(G, 90)
    mask = np.where(G >= threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def upper_boundary_mask(lower_mask: np.ndarray, img_gray: np.ndarray) -> np.ndarray:
    '''Build upper-boundary mask relative to lower boundary.
    For each column: brightest pixel ABOVE lowest nonzero row in lower_mask.
    Dilate result with (1,3). Source: Unet_Extraction.ipynb — ported verbatim.'''
    h, w = img_gray.shape
    upper_mask = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        nonzero_rows = np.nonzero(lower_mask[:, col])[0]
        if len(nonzero_rows) == 0:
            continue
        lower_boundary_row = nonzero_rows[-1]
        col_pixels = img_gray[:lower_boundary_row, col]
        if col_pixels.size == 0:
            continue
        upper_mask[int(np.argmax(col_pixels)), col] = 255
    kernel = np.ones((1, 3), np.uint8)
    return cv2.dilate(upper_mask, kernel)


def create_final_mask(img_path: str) -> np.ndarray:
    '''Combine lower + upper boundary masks. NOT a human annotation.
    Source: Unet_Extraction.ipynb — ported verbatim.'''
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')
    lower = lower_boundary_mask(img_gray)
    upper = upper_boundary_mask(lower, img_gray)
    return cv2.bitwise_or(lower, upper)


def column_fill(mask: np.ndarray) -> np.ndarray:
    '''Fill each column from first to last nonzero pixel with 255.
    Source: Unet_Extraction.ipynb — ported verbatim.'''
    filled = mask.copy()
    for col in range(filled.shape[1]):
        nonzero = np.nonzero(filled[:, col])[0]
        if len(nonzero) >= 2:
            filled[nonzero[0]:nonzero[-1] + 1, col] = 255
    return filled


def extract_bone_region(img: np.ndarray, mask: np.ndarray, size: int = 100) -> np.ndarray:
    '''Apply mask via bitwise_and and resize to (size, size).
    Source: Unet_Extraction.ipynb — ported verbatim.'''
    bone = cv2.bitwise_and(img, img, mask=mask)
    return cv2.resize(bone, (size, size))
