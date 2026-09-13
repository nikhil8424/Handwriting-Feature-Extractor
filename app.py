"""
Handwriting Feature Extractor - Streamlit Application
A simple tool to extract measurable handwriting features from images.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# Try importing cv2 with fallback
try:
    # Set environment variable to avoid GL dependencies
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    import cv2
except ImportError as e:
    st.error(f"❌ OpenCV (cv2) import failed: {str(e)}")
    st.error("This may be due to missing system libraries. The app requires OpenCV for image processing.")
    st.info("Please check the deployment logs and ensure system dependencies are installed.")
    st.stop()

from ocr import process_image, validate_image
from feature_extractor import extract_all_features
from csv_manager import (
    initialize_csv,
    load_features,
    append_features,
    get_sample_count,
    get_csv_bytes
)


# Page configuration
st.set_page_config(
    page_title="Handwriting Feature Extractor",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .feature-value {
        font-weight: bold;
        color: #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Initialize CSV on first run
    initialize_csv()
    
    # Main header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("✍️ Handwriting Feature Extractor")
    st.markdown("Extract measurable handwriting features from images using computer vision")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display current sample count
    sample_count = get_sample_count()
    st.info(f"📊 Samples currently stored: {sample_count}")
    
    st.markdown("---")
    
    # File upload section
    st.subheader("📤 Upload Handwriting Image")
    
    uploaded_file = st.file_uploader(
        "Choose a handwriting image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        try:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("❌ Failed to read the uploaded image. Please try a different file.")
                return
                
            # Validate image
            is_valid, error_msg = validate_image(image)
            if not is_valid:
                st.error(f"❌ Invalid image: {error_msg}")
                return
                
            # Display image preview
            st.subheader("👁️ Image Preview")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            
            # Extract features button
            st.markdown("---")
            if st.button("🚀 Extract Features", type="primary", use_container_width=True):
                with st.spinner("Processing image and extracting features..."):
                    try:
                        # Process image
                        processed_data = process_image(image)
                        
                        # Extract features
                        features = extract_all_features(processed_data)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📋 Extracted Features")
                        
                        # Create feature display
                        feature_display = {
                            "Slant": f"{features['slant']}°" if not np.isnan(features['slant']) else "N/A",
                            "Baseline Angle": f"{features['baseline_angle']}°" if not np.isnan(features['baseline_angle']) else "N/A",
                            "Average Size": f"{features['avg_size']} px" if not np.isnan(features['avg_size']) else "N/A",
                            "Letter Spacing": f"{features['letter_spacing']} px" if not np.isnan(features['letter_spacing']) else "N/A",
                            "Word Spacing": f"{features['word_spacing']} px" if not np.isnan(features['word_spacing']) else "N/A",
                            "Line Spacing": f"{features['line_spacing']} px" if not np.isnan(features['line_spacing']) else "N/A",
                            "Left Margin": f"{features['left_margin']} px" if not np.isnan(features['left_margin']) else "N/A",
                            "Right Margin": f"{features['right_margin']} px" if not np.isnan(features['right_margin']) else "N/A",
                            "Top Margin": f"{features['top_margin']} px" if not np.isnan(features['top_margin']) else "N/A",
                            "Bottom Margin": f"{features['bottom_margin']} px" if not np.isnan(features['bottom_margin']) else "N/A",
                            "Stroke Thickness": f"{features['stroke_thickness']} px" if not np.isnan(features['stroke_thickness']) else "N/A",
                            "Ink Density": f"{features['ink_density']}" if not np.isnan(features['ink_density']) else "N/A"
                        }
                        
                        # Display as a clean table
                        for feature_name, value in feature_display.items():
                            st.markdown(f"**{feature_name}:** <span class='feature-value'>{value}</span>", unsafe_allow_html=True)
                        
                        # Store features in session state for saving
                        st.session_state['current_features'] = features
                        st.session_state['current_image_name'] = uploaded_file.name
                        
                        st.success("✅ Features extracted successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during feature extraction: {str(e)}")
                        st.error("The image may not contain clear handwriting or may be of poor quality.")
                        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    
    # Save to CSV section (only show if features are extracted)
    if 'current_features' in st.session_state and 'current_image_name' in st.session_state:
        st.markdown("---")
        st.subheader("💾 Save to CSV")
        
        if st.button("Save Features to CSV", type="secondary", use_container_width=True):
            try:
                success = append_features(
                    st.session_state['current_image_name'],
                    st.session_state['current_features']
                )
                
                if success:
                    st.success(f"✅ Features saved for '{st.session_state['current_image_name']}'")
                    
                    # Update sample count
                    new_count = get_sample_count()
                    st.info(f"📊 Total samples stored: {new_count}")
                    
                    # Clear session state
                    del st.session_state['current_features']
                    del st.session_state['current_image_name']
                    
                    # Rerun to update UI
                    st.rerun()
                else:
                    st.error("❌ Failed to save features to CSV")
                    
            except Exception as e:
                st.error(f"❌ Error saving to CSV: {str(e)}")
    
    # Download CSV section
    st.markdown("---")
    st.subheader("📥 Download CSV")
    
    csv_bytes = get_csv_bytes()
    if csv_bytes and get_sample_count() > 0:
        st.download_button(
            label="Download handwriting_features.csv",
            data=csv_bytes,
            file_name="handwriting_features.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No samples stored yet. Upload and extract features first.")
    
    # Show existing data preview
    if get_sample_count() > 0:
        st.markdown("---")
        st.subheader("📊 Stored Data Preview")
        
        try:
            df = load_features()
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data preview: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p><strong>Note:</strong> This tool extracts measurable handwriting features using computer vision techniques.
        Features may not be accurately detected for poor-quality images or very stylized handwriting.</p>
        <p>Features with "N/A" could not be reliably detected from the image.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
