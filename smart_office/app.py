#!/usr/bin/env python3
"""
Smart Office Challenge - Streamlit Dashboard
Interactive web app for office object detection
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import tempfile
import time

# Configure page
st.set_page_config(
    page_title="Smart Office Challenge",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .detection-info {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def validate_image(uploaded_file):
    """
    Validate and safely load an uploaded image file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (PIL.Image object or None, error_message or None)
    """
    if uploaded_file is None:
        return None, "No file uploaded"
    
    try:
        # Check file size (optional: prevent extremely large files)
        file_size = uploaded_file.size
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return None, "File too large (max 50MB allowed)"
        
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Try to open with PIL
        image = Image.open(uploaded_file)
        
        # Verify it's actually an image by trying to load it
        image.verify()
        
        # Reopen the file since verify() closes it
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed (handles various formats)
        if image.mode != 'RGB':
            try:
                image = image.convert('RGB')
            except Exception as conv_error:
                return None, f"Failed to convert image to RGB: {str(conv_error)}"
        
        # Additional validation: check dimensions
        width, height = image.size
        if width < 10 or height < 10:
            return None, "Image too small (minimum 10x10 pixels)"
        
        if width > 10000 or height > 10000:
            return None, "Image too large (maximum 10000x10000 pixels)"
        
        return image, None
        
    except Exception as e:
        error_msg = f"Invalid image file: {str(e)}"
        return None, error_msg

class OfficeDetectionApp:
    def __init__(self):
        self.class_names = ['person', 'chair', 'monitor', 'keyboard', 'laptop', 'phone']
        self.class_colors = {
            'person': '#FF6B6B',
            'chair': '#4ECDC4',
            'monitor': '#45B7D1',
            'keyboard': '#96CEB4',
            'laptop': '#FFEAA7',
            'phone': '#DDA0DD'
        }

    def load_model(self, model_path):
        """Load YOLO model and store in session state"""
        try:
            model = YOLO(model_path)
            # Store model and status in session state
            st.session_state.yolo_model = model
            st.session_state.model_loaded = True
            st.session_state.model_path = model_path
            return True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.session_state.model_loaded = False
            return False

    def detect_objects(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """Run object detection on image"""
        if not st.session_state.get('model_loaded', False):
            return None, "Model not loaded"

        try:
            # Get model from session state
            model = st.session_state.yolo_model
            
            # Run inference
            results = model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                device='cpu',
                verbose=False
            )
            return results[0], None
        except Exception as e:
            return None, str(e)

    def draw_detections(self, image, results):
        """Draw bounding boxes and labels on image"""
        if results is None or results.boxes is None:
            return image, []

        # Convert to PIL for drawing
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image.copy()

        draw = ImageDraw.Draw(pil_image)

        # Try to load a font (Python 3.10 compatible)
        try:
            font_paths = [
                "arial.ttf",
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "C:/Windows/Fonts/arial.ttf"  # Windows
            ]

            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 24)
                    break
                except (IOError, OSError):
                    continue

            if font is None:
                font = ImageFont.load_default()

        except Exception:
            font = ImageFont.load_default()

        detections = []

        # Process each detection
        boxes = results.boxes
        for i in range(len(boxes)):
            # Get box coordinates
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())

            if cls < len(self.class_names):
                class_name = self.class_names[cls]
                color = self.class_colors.get(class_name, '#FF0000')

                # Draw bounding box
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Draw label
                label = f"{class_name}: {conf:.2f}"

                # Get text size for background
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Draw label background
                draw.rectangle([x1, y1-text_height-10, x1+text_width+10, y1],
                               fill=color, outline=color)

                # Draw text
                draw.text((x1+5, y1-text_height-5), label, fill='white', font=font)

                # Store detection info
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'area': int((x2-x1) * (y2-y1))
                })

        return pil_image, detections

    def create_confidence_chart(self, detections):
        """Create confidence score visualization"""
        if not detections:
            return None

        # Prepare data for plotting
        classes = [d['class'].title() for d in detections]
        confidences = [d['confidence'] for d in detections]
        colors = [self.class_colors.get(d['class'], '#FF0000') for d in detections]

        # Create bar chart
        fig = px.bar(
            x=classes,
            y=confidences,
            color=classes,
            title="Detection Confidence Scores",
            labels={'x': 'Object Class', 'y': 'Confidence Score'},
            color_discrete_map={cls.title(): self.class_colors.get(cls.lower(), '#FF0000')
                                for cls in set(classes)}
        )

        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_tickangle=-45
        )

        return fig

    def create_detection_summary(self, detections):
        """Create summary of detections"""
        if not detections:
            return None

        # Count objects by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Create summary dataframe
        summary_data = []
        for class_name, count in class_counts.items():
            avg_conf = np.mean([d['confidence'] for d in detections if d['class'] == class_name])
            summary_data.append({
                'Object': class_name.title(),
                'Count': count,
                'Avg Confidence': f"{avg_conf:.3f}",
                'Color': self.class_colors.get(class_name, '#FF0000')
            })

        return pd.DataFrame(summary_data)

    def create_detection_pie_chart(self, detections):
        """Create pie chart of object distribution"""
        if not detections:
            return None

        # Count objects by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class'].title()
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if not class_counts:
            return None

        # Create pie chart
        fig = px.pie(
            values=list(class_counts.values()),
            names=list(class_counts.keys()),
            title="Object Distribution",
            color_discrete_map={cls.title(): self.class_colors.get(cls.lower(), '#FF0000')
                                for cls in class_counts.keys()}
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)

        return fig

def main():
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'yolo_model' not in st.session_state:
        st.session_state.yolo_model = None
    if 'temp_model_path' not in st.session_state:
        st.session_state.temp_model_path = None

    # Initialize app (this won't affect model state anymore)
    app = OfficeDetectionApp()

    # Auto-load model if it exists and not already loaded
    auto_model_path = "best_office_model.pt"
    if not st.session_state.model_loaded and os.path.exists(auto_model_path):
        with st.spinner("Auto-loading model..."):
            success = app.load_model(auto_model_path)
            if success:
                st.success("✅ Auto-loaded best_office_model.pt successfully!")
            else:
                st.warning("⚠️ Failed to auto-load model, please upload manually.")

    # Main header
    st.markdown('<h1 class="main-header">🏢 Smart Office Challenge</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Office Object Detection System</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        st.subheader("Model Settings")
        
        # Show auto-load status
        auto_model_path = "best_office_model.pt"
        if os.path.exists(auto_model_path):
            st.info(f"🤖 Auto-detected: {auto_model_path}")
        else:
            st.warning("⚠️ best_office_model.pt not found - use manual upload")
        
        # Option 1: File path input (fallback)
        model_path_input = st.text_input("Manual Model Path", value="", help="Path to trained YOLO model (if not using auto-load)")
        
        # Option 2: File uploader (fallback)
        uploaded_model = st.file_uploader("Upload YOLOv8 Model (.pt)", type=["pt"], help="Upload if auto-load failed")
        
        # Determine which model to use (priority: auto-load > uploaded > manual path)
        model_path = None
        if st.session_state.model_loaded and 'model_path' in st.session_state:
            # Model already loaded
            model_path = st.session_state.model_path
            st.success(f"✅ Currently loaded: {os.path.basename(model_path)}")
        elif uploaded_model is not None:
            # Save uploaded file temporarily
            if st.session_state.temp_model_path is None or not os.path.exists(st.session_state.temp_model_path):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model:
                    temp_model.write(uploaded_model.read())
                    st.session_state.temp_model_path = temp_model.name
            model_path = st.session_state.temp_model_path
            st.success("✅ Model uploaded and ready to load!")
        elif model_path_input and os.path.exists(model_path_input):
            model_path = model_path_input
            st.info(f"✅ Manual path ready: {os.path.basename(model_path)}")
        elif not st.session_state.model_loaded:
            st.warning("⚠️ No model available - upload one above")

        # Load model button (only show if model not loaded and path available)
        if model_path and not st.session_state.model_loaded and st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                success = app.load_model(model_path)
                if success:
                    st.success("✅ Model loaded successfully!")
                    st.rerun()  # Refresh to update the UI
                else:
                    st.error("❌ Failed to load model")

        # Detection parameters
        st.subheader("Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05,
                                  help="Minimum confidence score for detections")
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05,
                                 help="IoU threshold for Non-Maximum Suppression")

        # Model info
        st.subheader("📊 Model Information")
        if st.session_state.model_loaded:
            st.success("🟢 Model Status: Loaded")
            if 'model_path' in st.session_state:
                st.info(f"📁 Model: {os.path.basename(st.session_state.model_path)}")
            st.info(f"🎯 Classes: {len(app.class_names)}")
        else:
            st.warning("🟡 Model Status: Not Loaded")

        # Debug info
        st.markdown("---")
        st.markdown("**Debug Info:**")
        st.write(f"[DEBUG] model_loaded = {st.session_state.model_loaded}")
        if st.session_state.temp_model_path:
            st.write(f"[DEBUG] temp_model_path = {st.session_state.temp_model_path}")

        # Class legend
        st.subheader("🏷️ Object Classes")
        for class_name in app.class_names:
            color = app.class_colors[class_name]
            st.markdown(
                f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                f'<div style="width: 20px; height: 20px; background-color: {color}; '
                f'margin-right: 10px; border-radius: 3px;"></div>'
                f'<span>{class_name.title()}</span></div>',
                unsafe_allow_html=True
            )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📷 Image Upload & Detection")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an office image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing office objects"
        )

        # FIXED: Robust image handling with validation
        if uploaded_file is not None:
            # Validate and load the image safely
            image, error_message = validate_image(uploaded_file)
            
            if error_message:
                # Display error message for invalid images
                st.error(f"❌ Image Error: {error_message}")
                st.info("Please upload a valid image file (PNG, JPG, JPEG, BMP, or TIFF)")
            elif image is not None:
                # Only display and process if image is valid
                st.subheader("📸 Original Image")
                try:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                except Exception as display_error:
                    st.error(f"❌ Failed to display image: {str(display_error)}")
                    image = None  # Reset image to prevent further processing

                # Run detection only if image is valid
                if image is not None and st.session_state.model_loaded:
                    if st.button("🔍 Run Detection", type="primary"):
                        with st.spinner("Detecting objects..."):
                            try:
                                # Convert PIL to numpy array for YOLO
                                img_array = np.array(image)

                                # Run detection
                                results, error = app.detect_objects(
                                    img_array,
                                    conf_threshold=conf_threshold,
                                    iou_threshold=iou_threshold
                                )

                                if error:
                                    st.error(f"Detection failed: {error}")
                                elif results is not None:
                                    # Draw detections
                                    detected_image, detections = app.draw_detections(image, results)

                                    # Display results
                                    st.subheader("🎯 Detection Results")
                                    try:
                                        st.image(detected_image, caption="Detected Objects", use_container_width=True)
                                        
                                        # Store results in session state
                                        st.session_state.detections = detections
                                        st.session_state.detected_image = detected_image
                                        
                                        if detections:
                                            st.success(f"✅ Detected {len(detections)} objects!")
                                        else:
                                            st.info("ℹ️ No objects detected with current confidence threshold.")
                                    except Exception as result_error:
                                        st.error(f"❌ Failed to display detection results: {str(result_error)}")
                                else:
                                    st.warning("No objects detected in the image.")
                                    
                            except Exception as detection_error:
                                st.error(f"❌ Detection process failed: {str(detection_error)}")
                                
                elif image is not None and not st.session_state.model_loaded:
                    st.warning("⚠️ Please load a model first using the sidebar.")
            else:
                st.warning("⚠️ Failed to load the uploaded image. Please try a different file.")

    with col2:
        st.header("📊 Detection Analytics")

        # Check if we have detection results
        if hasattr(st.session_state, 'detections') and st.session_state.detections:
            detections = st.session_state.detections

            # Summary metrics
            st.subheader("📈 Summary")
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Total Objects", len(detections))

            with col_b:
                avg_conf = np.mean([d['confidence'] for d in detections])
                st.metric("Avg Confidence", f"{avg_conf:.3f}")

            # Detection summary table
            summary_df = app.create_detection_summary(detections)
            if summary_df is not None:
                st.subheader("🏷️ Object Summary")
                st.dataframe(
                    summary_df[['Object', 'Count', 'Avg Confidence']],
                    use_container_width=True, hide_index=True
                )

            # Confidence chart
            conf_chart = app.create_confidence_chart(detections)
            if conf_chart:
                st.plotly_chart(conf_chart, use_container_width=True)

            # Object distribution pie chart
            pie_chart = app.create_detection_pie_chart(detections)
            if pie_chart:
                st.plotly_chart(pie_chart, use_container_width=True)

            # Detailed detection info
            with st.expander("🔍 Detailed Detection Info"):
                for i, detection in enumerate(detections):
                    st.markdown(f"""
                    <div class="detection-info">
                        <strong>Detection {i+1}:</strong><br>
                        🏷️ Class: {detection['class'].title()}<br>
                        📊 Confidence: {detection['confidence']:.3f}<br>
                        📐 Bounding Box: {detection['bbox']}<br>
                        📏 Area: {detection['area']} pixels
                    </div>
                    """, unsafe_allow_html=True)

            # Download results
            # st.subheader("💾 Export Results")

            # # Export detection data as JSON
            # if st.button("📄 Download Detection Data"):
            #     export_data = {
            #         'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            #         'model_path': st.session_state.get('model_path', 'Unknown'),
            #         'parameters': {
            #             'confidence_threshold': conf_threshold,
            #             'iou_threshold': iou_threshold
            #         },
            #         'detections': detections,
            #         'summary': {
            #             'total_objects': len(detections),
            #             'average_confidence': float(np.mean([d['confidence'] for d in detections])),
            #             'class_distribution': dict(summary_df.set_index('Object')['Count'])
            #             if summary_df is not None else {}
            #         }
            #     }

            #     st.download_button(
            #         label="📥 Download JSON",
            #         data=json.dumps(export_data, indent=2),
            #         file_name=f"detection_results_{int(time.time())}.json",
            #         mime="application/json"
            #     )
        else:
            st.info("👆 Upload an image and run detection to see analytics.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🏢 <strong>Smart Office Challenge</strong> | 
        Built with ❤️ using Streamlit & YOLOv8 | 
        📧 AI-Powered Object Detection System
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
