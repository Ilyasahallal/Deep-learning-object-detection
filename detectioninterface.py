import streamlit as st
import cv2
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
from PIL import Image

class PredictionConfig(Config):
    NAME = "kangaroo_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_detection_model():
    # Create config
    cfg = PredictionConfig()
    # Create model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # Load weights
    model.load_weights('mask_rcnn_kangaroo_cfg_0001.h5', by_name=True)
    return model

def process_image(image, model):
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    
    # If image is RGBA, convert to RGB
    if image_array.shape[-1] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Prepare image for model
    scaled_image = mold_image(image_array, model.config)
    sample = np.expand_dims(scaled_image, 0)
    
    # Make prediction
    results = model.detect(sample, verbose=0)[0]
    
    return results, image_array

def draw_detections(image_array, results):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image_array)
    
    # Draw each detected box
    for box in results['rois']:
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, 
                        fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        
        # Add confidence score if available
        if 'scores' in results:
            score = results['scores'][list(results['rois']).index(box)]
            ax.text(x1, y1-10, f'Kangaroo: {score:.2f}', 
                   color='red', fontsize=12, backgroundcolor='white')
    
    # Remove axes
    ax.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    return buf

def main():
    st.title("Kangaroo Detection System")
    st.write("Upload an image to detect kangaroos!")
    
    # Model loading with status
    with st.spinner('Loading detection model...'):
        model = load_detection_model()
    st.success('Model loaded successfully!')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Process image and get detections
        with st.spinner('Detecting kangaroos...'):
            results, image_array = process_image(image, model)
        
        # Display results
        if len(results['rois']) > 0:
            st.subheader(f"Found {len(results['rois'])} kangaroo(s)!")
            
            # Draw detections
            detection_image = draw_detections(image_array, results)
            
            # Display image with detections
            st.subheader("Detection Results")
            st.image(detection_image, use_column_width=True)
            
            # Display confidence scores
            if 'scores' in results:
                st.subheader("Detection Confidence Scores")
                for i, score in enumerate(results['scores'], 1):
                    st.write(f"Kangaroo {i}: {score:.2%} confidence")
        else:
            st.info("No kangaroos detected in this image.")

if __name__ == '__main__':
    main()