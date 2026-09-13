"""
OCR and Image Preprocessing Module
Handles image preprocessing, text detection, and layout analysis.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple


def preprocess_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Preprocess handwriting image for feature extraction.
    
    Steps:
    1. Convert to grayscale
    2. Apply noise reduction (bilateral filter)
    3. Enhance contrast (CLAHE)
    4. Binarize using adaptive thresholding
    5. Clean up noise with morphological operations
    
    Args:
        image: Input image (BGR or RGB)
        
    Returns:
        Dictionary containing preprocessed image data
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Noise reduction using bilateral filter (preserves edges)
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=10
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_noise)
        
        return {
            'original': image,
            'gray': gray,
            'denoised': denoised,
            'enhanced': enhanced,
            'binary': cleaned,
            'height': image.shape[0],
            'width': image.shape[1]
        }
        
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")


def detect_contours(binary_img: np.ndarray) -> List[np.ndarray]:
    """
    Detect connected components (contours) in binary image.
    
    Args:
        binary_img: Binary image (white ink on black background)
        
    Returns:
        List of contours
    """
    try:
        # Find external contours
        contours, _ = cv2.findContours(
            binary_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size to remove noise
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Keep reasonable-sized components
            if w >= 3 and h >= 5 and w * h >= 20:
                filtered_contours.append(contour)
                
        return filtered_contours
        
    except Exception:
        return []


def detect_text_lines(binary_img: np.ndarray, contours: List) -> List[Dict[str, Any]]:
    """
    Detect text lines using morphological operations and contour analysis.
    
    Method:
    1. Use horizontal dilation to connect components within lines
    2. Find line regions
    3. Extract bounding boxes for each line
    
    Args:
        binary_img: Binary image
        contours: List of detected contours
        
    Returns:
        List of line dictionaries with bounding box information
    """
    try:
        if not contours:
            return []
            
        img_height, img_width = binary_img.shape[:2]
        
        # Horizontal dilation to connect components within lines
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(20, img_width // 10), 1)
        )
        dilated_horizontal = cv2.dilate(binary_img, horizontal_kernel, iterations=2)
        
        # Find line contours
        line_contours, _ = cv2.findContours(
            dilated_horizontal,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract line bounding boxes
        lines = []
        for line_contour in line_contours:
            x, y, w, h = cv2.boundingRect(line_contour)
            
            # Filter lines by reasonable dimensions
            if w >= 30 and h >= 10 and h <= img_height * 0.3:
                lines.append({
                    'bbox': [x, y, w, h],
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                })
                
        # Sort lines by vertical position (top to bottom)
        lines.sort(key=lambda l: l['y'])
        
        return lines
        
    except Exception:
        return []


def process_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Complete image processing pipeline.
    
    Args:
        image: Input image (BGR or RGB)
        
    Returns:
        Dictionary containing all processed data needed for feature extraction
    """
    try:
        # Preprocess image
        preprocessed = preprocess_image(image)
        
        # Detect contours (connected components)
        contours = detect_contours(preprocessed['binary'])
        
        # Detect text lines
        lines = detect_text_lines(preprocessed['binary'], contours)
        
        # Add contours and lines to preprocessed data
        preprocessed['contours'] = contours
        preprocessed['lines'] = lines
        
        return preprocessed
        
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that image is suitable for processing.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
        
    if len(image.shape) not in [2, 3]:
        return False, "Invalid image dimensions"
        
    if len(image.shape) == 3 and image.shape[2] not in [3, 4]:
        return False, "Invalid number of color channels"
        
    height, width = image.shape[:2]
    
    if height < 50 or width < 50:
        return False, "Image too small (minimum 50x50 pixels)"
        
    if height > 5000 or width > 5000:
        return False, "Image too large (maximum 5000x5000 pixels)"
        
    return True, ""
