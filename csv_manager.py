"""
CSV Manager Module
Handles all CSV operations for storing and retrieving handwriting features.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, Optional


CSV_FILENAME = "handwriting_features.csv"
CSV_COLUMNS = [
    'image_name',
    'timestamp',
    'slant',
    'baseline_angle',
    'avg_size',
    'letter_spacing',
    'word_spacing',
    'line_spacing',
    'left_margin',
    'right_margin',
    'top_margin',
    'bottom_margin',
    'stroke_thickness',
    'ink_density'
]


def initialize_csv() -> None:
    """
    Initialize CSV file with headers if it doesn't exist.
    """
    if not os.path.exists(CSV_FILENAME):
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(CSV_FILENAME, index=False)
        print(f"Created new CSV file: {CSV_FILENAME}")


def load_features() -> pd.DataFrame:
    """
    Load existing features from CSV file.
    
    Returns:
        DataFrame containing all stored features, or empty DataFrame if file doesn't exist
    """
    try:
        if os.path.exists(CSV_FILENAME):
            df = pd.read_csv(CSV_FILENAME)
            return df
        else:
            return pd.DataFrame(columns=CSV_COLUMNS)
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return pd.DataFrame(columns=CSV_COLUMNS)


def append_features(image_name: str, features: Dict[str, float]) -> bool:
    """
    Append extracted features as a new row to the CSV file.
    
    Args:
        image_name: Name of the image file
        features: Dictionary of extracted features
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create row data with correct column order
        row_data = {
            'image_name': image_name,
            'timestamp': timestamp,
            'slant': features.get('slant', ''),
            'baseline_angle': features.get('baseline_angle', ''),
            'avg_size': features.get('avg_size', ''),
            'letter_spacing': features.get('letter_spacing', ''),
            'word_spacing': features.get('word_spacing', ''),
            'line_spacing': features.get('line_spacing', ''),
            'left_margin': features.get('left_margin', ''),
            'right_margin': features.get('right_margin', ''),
            'top_margin': features.get('top_margin', ''),
            'bottom_margin': features.get('bottom_margin', ''),
            'stroke_thickness': features.get('stroke_thickness', ''),
            'ink_density': features.get('ink_density', '')
        }
        
        # Load existing data
        df = load_features()
        
        # Append new row
        new_row = pd.DataFrame([row_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(CSV_FILENAME, index=False)
        
        return True
        
    except Exception as e:
        print(f"Error appending features: {str(e)}")
        return False


def get_sample_count() -> int:
    """
    Get the number of samples currently stored in the CSV.
    
    Returns:
        Number of samples (rows) in the CSV
    """
    try:
        df = load_features()
        return len(df)
    except Exception:
        return 0


def get_csv_bytes() -> Optional[bytes]:
    """
    Get CSV file as bytes for download.
    
    Returns:
        CSV file content as bytes, or None if error occurs
    """
    try:
        if os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, 'rb') as f:
                return f.read()
        return None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None


def clear_csv() -> bool:
    """
    Clear all data from CSV file (keeping headers).
    
    Returns:
        True if successful, False otherwise
    """
    try:
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(CSV_FILENAME, index=False)
        return True
    except Exception as e:
        print(f"Error clearing CSV: {str(e)}")
        return False


def get_feature_summary() -> Dict[str, Any]:
    """
    Get summary statistics of stored features.
    
    Returns:
        Dictionary with summary statistics
    """
    try:
        df = load_features()
        
        if len(df) == 0:
            return {
                'total_samples': 0,
                'numeric_columns': []
            }
            
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary = {
            'total_samples': len(df),
            'numeric_columns': numeric_cols,
            'column_stats': {}
        }
        
        # Calculate basic stats for numeric columns
        for col in numeric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                summary['column_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'count': int(df[col].notna().sum())
                }
                
        return summary
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return {
            'total_samples': 0,
            'numeric_columns': []
        }


# Import numpy for summary statistics
import numpy as np
