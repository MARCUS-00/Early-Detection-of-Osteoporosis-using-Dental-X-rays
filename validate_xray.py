import cv2
import numpy as np


def validate_dental_xray(img_path: str) -> tuple:
    '''
    Validate that the uploaded image is likely a dental X-ray.

    Checks:
      1. Minimum size (too small = not a real X-ray)
      2. Grayscale check (X-rays are grayscale; colored images are rejected)
      3. Not a pure black/white blank image

    Returns:
        (is_valid: bool, reason: str)
        reason is a user-friendly message when invalid, empty string when valid.
    '''
    img = cv2.imread(str(img_path))
    if img is None:
        return False, 'Could not read the uploaded file. Please upload a valid image.'

    h, w = img.shape[:2]

    # Check 1: Minimum size
    if w < 100 or h < 80:
        return False, (
            f'Image is too small ({w}x{h} px). '
            'Please upload a full dental panoramic X-ray.'
        )

    # Check 2: Grayscale (dental X-rays are grayscale)
    img_f = img.astype(np.float32)
    colorfulness = (img_f.max(axis=2) - img_f.min(axis=2)).mean()
    if colorfulness > 30:
        return False, (
            'The uploaded image appears to be a colour photograph or graphic '
            f'(colour score: {colorfulness:.0f}/30 max allowed). '
            'Please upload a grayscale dental panoramic X-ray.'
        )

    # Check 3: Not a blank / solid-colour image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_dev = float(gray.std())
    if std_dev < 5:
        return False, (
            'The uploaded image appears blank or nearly uniform. '
            'Please upload a dental panoramic X-ray with visible bone structure.'
        )

    return True, ''
