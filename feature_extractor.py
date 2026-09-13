"""
Handwriting Feature Extractor Module
Extracts measurable handwriting features using computer vision techniques.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from scipy import ndimage


def extract_slant(binary_img: np.ndarray, gray_img: np.ndarray) -> float:
    """
    Estimate dominant handwriting slant angle using Sobel gradients and PCA.
    
    Method:
    1. Apply Sobel filters to detect edge orientations
    2. Use gradient directions to estimate stroke angles
    3. Compute dominant angle using histogram analysis
    
    Returns:
        Slant angle in degrees (positive = right slant, negative = left slant)
    """
    try:
        # Sobel gradients
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient angles
        angles = np.arctan2(sobel_y, sobel_x)
        
        # Mask to ink regions
        ink_mask = binary_img > 0
        valid_angles = angles[ink_mask]
        
        if len(valid_angles) < 100:
            return np.nan
            
        # Convert to degrees and filter for vertical-ish strokes (45-135 degrees)
        angle_degrees = np.degrees(valid_angles)
        vertical_strokes = angle_degrees[(angle_degrees >= 45) & (angle_degrees <= 135)]
        
        if len(vertical_strokes) < 50:
            return np.nan
            
        # Compute histogram and find dominant angle
        hist, bins = np.histogram(vertical_strokes, bins=30, range=(45, 135))
        dominant_bin = np.argmax(hist)
        dominant_angle = bins[dominant_bin]
        
        # Convert to slant relative to vertical (90 degrees)
        slant = dominant_angle - 90.0
        
        return round(float(slant), 1)
        
    except Exception:
        return np.nan


def extract_baseline_angle(lines: List[Dict]) -> float:
    """
    Estimate baseline angle from detected text lines.
    
    Method:
    1. Fit linear regression to each line's bounding box
    2. Compute average slope across all lines
    3. Convert slope to angle in degrees
    
    Args:
        lines: List of detected line bounding boxes [{'bbox': [x, y, w, h]}, ...]
    
    Returns:
        Baseline angle in degrees (positive = ascending, negative = descending)
    """
    try:
        if len(lines) < 2:
            return np.nan
            
        # Extract line centers
        centers = []
        for line in lines:
            x, y, w, h = line['bbox']
            center_x = x + w / 2
            center_y = y + h / 2
            centers.append([center_x, center_y])
            
        centers = np.array(centers)
        
        # Linear regression to find overall trend
        if len(centers) < 2:
            return np.nan
            
        # Fit line: y = mx + b
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]
        
        # Calculate slope using least squares
        slope = np.polyfit(x_coords, y_coords, 1)[0]
        
        # Convert slope to angle (in degrees)
        angle = np.degrees(np.arctan(slope))
        
        return round(float(angle), 1)
        
    except Exception:
        return np.nan


def extract_size(binary_img: np.ndarray, contours: List) -> float:
    """
    Estimate average character size from connected components.
    
    Method:
    1. Analyze all detected contours
    2. Filter for reasonable character-sized components
    3. Compute median height as representative size
    
    Returns:
        Average character height in pixels
    """
    try:
        heights = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter for character-sized components (not too small, not too large)
            if 8 <= h <= 100 and 4 <= w <= 80:
                heights.append(h)
                
        if len(heights) < 5:
            return np.nan
            
        # Use median to be robust against outliers
        avg_size = float(np.median(heights))
        
        return round(avg_size, 1)
        
    except Exception:
        return np.nan


def extract_spacing_metrics(binary_img: np.ndarray, lines: List[Dict], contours: List) -> Tuple[float, float, float]:
    """
    Extract letter spacing, word spacing, and line spacing.
    
    Methods:
    - Letter spacing: Horizontal gaps between adjacent components within lines
    - Word spacing: Larger horizontal gaps between component groups
    - Line spacing: Vertical distance between line baselines
    
    Returns:
        Tuple of (letter_spacing, word_spacing, line_spacing) in pixels
    """
    try:
        # Letter spacing: gaps between adjacent components
        letter_gaps = []
        for line in lines:
            line_components = []
            x, y, w, h = line['bbox']
            
            # Find contours within this line
            for contour in contours:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                if y <= cy <= y + h:  # Contour is within this line
                    line_components.append((cx, cw))
                    
            # Sort by x position
            line_components.sort(key=lambda x: x[0])
            
            # Calculate gaps between adjacent components
            for i in range(len(line_components) - 1):
                curr_x, curr_w = line_components[i]
                next_x, next_w = line_components[i + 1]
                gap = next_x - (curr_x + curr_w)
                if gap > 0 and gap < 50:  # Reasonable letter spacing
                    letter_gaps.append(gap)
                    
        letter_spacing = float(np.median(letter_gaps)) if letter_gaps else np.nan
        
        # Word spacing: larger gaps (simplified as larger horizontal gaps)
        word_gaps = [gap for gap in letter_gaps if gap > 15]  # Threshold for word gaps
        word_spacing = float(np.median(word_gaps)) if word_gaps else np.nan
        
        # Line spacing: vertical distance between lines
        line_gaps = []
        if len(lines) >= 2:
            line_y_positions = [line['bbox'][1] for line in lines]
            line_y_positions.sort()
            
            for i in range(len(line_y_positions) - 1):
                gap = line_y_positions[i + 1] - line_y_positions[i]
                if gap > 0:
                    line_gaps.append(gap)
                    
        line_spacing = float(np.median(line_gaps)) if line_gaps else np.nan
        
        return (
            round(letter_spacing, 1) if not np.isnan(letter_spacing) else np.nan,
            round(word_spacing, 1) if not np.isnan(word_spacing) else np.nan,
            round(line_spacing, 1) if not np.isnan(line_spacing) else np.nan
        )
        
    except Exception:
        return np.nan, np.nan, np.nan


def extract_margins(binary_img: np.ndarray, contours: List) -> Dict[str, float]:
    """
    Calculate page margins relative to handwriting bounding box.
    
    Method:
    1. Find overall bounding box of all handwriting
    2. Calculate distances from image edges
    
    Returns:
        Dictionary with left, right, top, bottom margins in pixels
    """
    try:
        if not contours:
            return {
                'left': np.nan,
                'right': np.nan,
                'top': np.nan,
                'bottom': np.nan
            }
            
        img_height, img_width = binary_img.shape[:2]
        
        # Find overall bounding box of all contours
        all_points = np.vstack([cv2.boundingRect(c) for c in contours])
        min_x = int(np.min(all_points[:, 0]))
        max_x = int(np.max(all_points[:, 0] + all_points[:, 2]))
        min_y = int(np.min(all_points[:, 1]))
        max_y = int(np.max(all_points[:, 1] + all_points[:, 3]))
        
        # Calculate margins
        left_margin = min_x
        right_margin = img_width - max_x
        top_margin = min_y
        bottom_margin = img_height - max_y
        
        return {
            'left': float(left_margin),
            'right': float(right_margin),
            'top': float(top_margin),
            'bottom': float(bottom_margin)
        }
        
    except Exception:
        return {
            'left': np.nan,
            'right': np.nan,
            'top': np.nan,
            'bottom': np.nan
        }


def extract_stroke_thickness(binary_img: np.ndarray) -> float:
    """
    Estimate average stroke thickness using distance transform.
    
    Method:
    1. Apply distance transform to binary image
    2. Find maximum distance for each pixel (radius of stroke)
    3. Stroke thickness = 2 * radius (diameter)
    4. Use median of all stroke pixels
    
    Returns:
        Average stroke thickness in pixels
    """
    try:
        # Distance transform (gives distance to nearest background pixel)
        dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
        
        # Get distances only for ink pixels
        ink_distances = dist_transform[binary_img > 0]
        
        if len(ink_distances) < 10:
            return np.nan
            
        # Stroke thickness = 2 * distance (diameter)
        thicknesses = ink_distances * 2
        
        # Use median for robustness
        avg_thickness = float(np.median(thicknesses))
        
        return round(avg_thickness, 1)
        
    except Exception:
        return np.nan


def extract_ink_density(binary_img: np.ndarray) -> float:
    """
    Calculate ink density as ratio of ink pixels to total pixels.
    
    Method:
    1. Count ink pixels (foreground)
    2. Count total pixels in image
    3. Calculate ratio
    
    Returns:
        Ink density as a float between 0 and 1
    """
    try:
        total_pixels = binary_img.size
        ink_pixels = np.count_nonzero(binary_img)
        
        if total_pixels == 0:
            return np.nan
            
        density = ink_pixels / total_pixels
        
        return round(float(density), 3)
        
    except Exception:
        return np.nan


def extract_all_features(preprocessed_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract all handwriting features from preprocessed image data.
    
    Args:
        preprocessed_data: Dictionary containing processed image data from OCR module
        
    Returns:
        Dictionary with all extracted features
    """
    try:
        binary_img = preprocessed_data['binary']
        gray_img = preprocessed_data['gray']
        contours = preprocessed_data['contours']
        lines = preprocessed_data['lines']
        
        # Extract individual features
        slant = extract_slant(binary_img, gray_img)
        baseline_angle = extract_baseline_angle(lines)
        avg_size = extract_size(binary_img, contours)
        letter_spacing, word_spacing, line_spacing = extract_spacing_metrics(
            binary_img, lines, contours
        )
        margins = extract_margins(binary_img, contours)
        stroke_thickness = extract_stroke_thickness(binary_img)
        ink_density = extract_ink_density(binary_img)
        
        return {
            'slant': slant,
            'baseline_angle': baseline_angle,
            'avg_size': avg_size,
            'letter_spacing': letter_spacing,
            'word_spacing': word_spacing,
            'line_spacing': line_spacing,
            'left_margin': margins['left'],
            'right_margin': margins['right'],
            'top_margin': margins['top'],
            'bottom_margin': margins['bottom'],
            'stroke_thickness': stroke_thickness,
            'ink_density': ink_density
        }
        
    except Exception as e:
        # Return all NaN if extraction fails
        return {
            'slant': np.nan,
            'baseline_angle': np.nan,
            'avg_size': np.nan,
            'letter_spacing': np.nan,
            'word_spacing': np.nan,
            'line_spacing': np.nan,
            'left_margin': np.nan,
            'right_margin': np.nan,
            'top_margin': np.nan,
            'bottom_margin': np.nan,
            'stroke_thickness': np.nan,
            'ink_density': np.nan
        }
