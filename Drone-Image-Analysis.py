import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from tqdm import tqdm
from google import genai
from google.genai import types 
from PIL import Image



def visualize_segmented_image(label_image, color_map):
    """Convert a label map to an RGB image using a color map."""
    if label_image.ndim == 3 and label_image.shape[-1] == 1:
        label_image = label_image.squeeze(axis=-1)
    
    # Create a color mapping array for efficient conversion
    max_label = max(color_map.keys())
    color_array = np.zeros((max_label + 1, 3), dtype=np.uint8)
    
    for label, color in color_map.items():
        color_array[label] = color
    
    # Use direct indexing for faster conversion
    rgb_image = color_array[label_image]
    
    return rgb_image

def height_adaptive_sliding_window(image, model, color_map, 
                                  training_height=17.5, 
                                  capture_height=90,
                                  base_size=512, 
                                  overlap_ratio=0.5,
                                  batch_size=4,
                                  verbose=True):
    """
    Perform semantic segmentation using a height-adaptive sliding window approach.
    
    Args:
        image: Input RGB image captured at 'capture_height' meters
        model: Model trained on images captured at 'training_height' meters
        color_map: Dictionary mapping class indices to RGB colors
        training_height: Height (in meters) at which training images were captured
        capture_height: Height (in meters) at which the input image was captured
        base_size: Size to which patches are resized before prediction (model input size)
        overlap_ratio: Amount of overlap between adjacent windows (0-1)
        batch_size: Number of patches to process in a single batch
        verbose: Whether to print progress information
        
    Returns:
        Segmentation visualization (RGB) and class labels
    """
    # Calculate the scale factor between training and testing heights
    scale_factor = capture_height / training_height
    
    # Use a fixed window size
    window_size = base_size
    
    if verbose:
        print(f"Training height: {training_height}m, Capture height: {capture_height}m")
        print(f"Scale factor: {scale_factor}")
        print(f"Window size: {window_size}px")
    
    # Calculate stride based on overlap ratio
    stride = int(window_size * (1 - overlap_ratio))
    if stride < 1:
        stride = 1  # Ensure minimum stride of 1 pixel
    
    h, w, _ = image.shape
    
    if verbose:
        print(f"Image dimensions: {w}x{h}")
        print(f"Window size: {window_size}px, Stride: {stride}px")
    
    # Determine number of classes from model output
    dummy_input = np.zeros((1, base_size, base_size, 3), dtype=np.float32)
    num_classes = model.predict(dummy_input, verbose=0).shape[-1]
    
    # Initialize arrays for predictions and counting overlaps
    predictions = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w, 1), dtype=np.float32)
    
    # Generate all patch coordinates
    patch_coords = []
    
    # Adjust the range to cover the entire image including the edges
    # This ensures we have full coverage
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Make sure the window doesn't go beyond the image boundary
            actual_y = min(y, h - window_size) if y + window_size > h else y
            actual_x = min(x, w - window_size) if x + window_size > w else x
            patch_coords.append((actual_y, actual_x))
    
    if verbose:
        print(f"Processing {len(patch_coords)} windows with batch size {batch_size}...")
    
    start_time = time.time()
    
    # Process patches in batches
    batch_count = 0
    total_batches = (len(patch_coords) + batch_size - 1) // batch_size
    
    for i in range(0, len(patch_coords), batch_size):
        batch_coords = patch_coords[i:i+batch_size]
        batch_patches = []
        
        batch_count += 1
        if verbose and batch_count % 10 == 0:
            elapsed = time.time() - start_time
            estimated_total = (elapsed / batch_count) * total_batches
            remaining = max(0, estimated_total - elapsed)
            print(f"Processing batch {batch_count}/{total_batches} - ETA: {remaining:.1f}s")
        
        for y, x in batch_coords:
            # Extract window
            patch = image[y:y+window_size, x:x+window_size]
            
            # Resize to model input size and normalize
            patch_resized = cv2.resize(patch, (base_size, base_size))
            patch_normalized = patch_resized.astype(np.float32) / 255.0
            
            batch_patches.append(patch_normalized)
        
        # Predict on batch
        batch_input = np.array(batch_patches)
        batch_predictions = model.predict(batch_input, verbose=0)
        
        # Process batch results
        for j, (y, x) in enumerate(batch_coords):
            pred = batch_predictions[j]
            
            # Resize prediction back to window size if needed
            if base_size != window_size:
                pred_resized = np.zeros((window_size, window_size, num_classes), dtype=np.float32)
                for c in range(num_classes):
                    class_prob = pred[:, :, c]
                    class_prob_resized = cv2.resize(class_prob, (window_size, window_size), 
                                                interpolation=cv2.INTER_LINEAR)
                    pred_resized[:, :, c] = class_prob_resized
            else:
                pred_resized = pred
            
            # Add to prediction and count maps
            predictions[y:y+window_size, x:x+window_size] += pred_resized
            count_map[y:y+window_size, x:x+window_size, 0] += 1
    
    # Check if there are any uncovered areas
    uncovered = count_map == 0
    if np.any(uncovered):
        uncovered_count = np.sum(uncovered)
        if verbose:
            print(f"Warning: {uncovered_count} pixels not covered by any window")
        
        # Simple fix: set uncovered areas to have a count of 1 to avoid division by zero
        count_map[uncovered] = 1
        
        # Set the predictions for uncovered areas to the nearest predicted values
        # This is a simple approach - for a more sophisticated approach, you could
        # extract and process these regions separately
        if uncovered_count > 0:
            # Get indices of uncovered pixels
            y_indices, x_indices = np.where(uncovered[:,:,0])
            
            # For each uncovered pixel, find the nearest predicted pixel
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                
                # Simple approach: use the prediction from the nearest covered pixel
                # This is fast but may not be as accurate as re-running the model
                
                # Find the nearest non-zero count
                # Start with a small radius and increase if needed
                radius = 5
                found = False
                
                while not found and radius < 50:
                    # Define search boundaries
                    y_min = max(0, y - radius)
                    y_max = min(h, y + radius + 1)
                    x_min = max(0, x - radius)
                    x_max = min(w, x + radius + 1)
                    
                    # Extract the region from count_map
                    region = count_map[y_min:y_max, x_min:x_max, 0]
                    
                    # Find non-zero elements
                    non_zero_y, non_zero_x = np.where(region > 0)
                    
                    if len(non_zero_y) > 0:
                        # Calculate distance to each non-zero element
                        distances = (non_zero_y - (y - y_min))**2 + (non_zero_x - (x - x_min))**2
                        nearest_idx = np.argmin(distances)
                        
                        # Get coordinates of the nearest non-zero element
                        nearest_y = y_min + non_zero_y[nearest_idx]
                        nearest_x = x_min + non_zero_x[nearest_idx]
                        
                        # Copy the prediction from the nearest pixel
                        predictions[y, x] = predictions[nearest_y, nearest_x]
                        found = True
                    else:
                        # Increase search radius
                        radius *= 2
    
    # Average predictions by count
    final_prediction = predictions / count_map
    
    # Get class with highest probability for each pixel
    predicted_labels = np.argmax(final_prediction, axis=-1)
    
    # Convert to RGB visualization
    segmented_image = visualize_segmented_image(predicted_labels, color_map)
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Prediction completed in {elapsed_time:.2f} seconds")
        print(f"Final output shape: {segmented_image.shape}")
    
    return segmented_image, predicted_labels

def display_result(original_image, mask, save_path=None, figsize=(15, 8)):
    """Display and optionally save the original image and its segmentation mask."""
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('Original Image (90m height)')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.axis('off')
    plt.title('Predicted Segmentation Mask')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def calculate_real_world_area(pixel_count, image_height, reference_height=30.5, reference_gsd=0.05):
    """
    Calculate the real-world area based on the number of pixels and image height.
    """
    gsd = reference_gsd * (image_height / reference_height)
    return pixel_count * (gsd ** 2)

def compute_class_metrics(predicted_labels, class_names=None):
    """Compute class distribution in the predicted segmentation."""
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    total_pixels = predicted_labels.size

    #print("\nClass distribution:")
    for i, label in enumerate(unique_labels):
        percentage = (counts[i] / total_pixels) * 100
        if class_names and label < len(class_names):
            #print(f"Class {label} ({class_names[label]}): {percentage:.2f}%")
            st.write(f"{class_names[label]}: **{percentage:.2f} %**")
        else:
            #print(f"Class {label}: {percentage:.2f}%")
            st.write(f"Class {label}: **{percentage:.2f} %**")

    # Create a list to store class distribution text
    class_distribution_text = []
    class_distribution_text.append("Class distribution:\n")  # Add the header
    for i, label in enumerate(unique_labels):
        percentage = (counts[i] / total_pixels) * 100
        if class_names and label < len(class_names):
            class_distribution_text.append(f"{class_names[label]}: {percentage:.2f} %\n")
        else:
            class_distribution_text.append(f"Class {label}: {percentage:.2f} %\n")

    # Join the list into a single string
    class_distribution_info = "".join(class_distribution_text)
    # print("text", class_distribution_info)
    return class_distribution_info
    

def llm_analysis(class_distribution_text, image_height):
    api_key = st.secrets["api_keys"]["genai"]
    client = genai.Client(api_key=api_key)

    llm_prompt = f"""
        Drone Image Segmentation Analysis
        This image was captured at a height of {image_height} and processed using a deep learning model.
        {class_distribution_text}
        Analysis Required:
        1. Identify the dominant land cover.
        2. Detect potential issues related to land use (e.g., deforestation, urban expansion).
        3. Assess the presence of people in the scene and highlight potential risks.
        4. Provide insights on flood risk areas based on terrain distribution.
        5. Suggest environmental conservation measures.
        Make the analysis short and concise, focusing on the most relevant aspects of the segmentation results.
        """

    response = client.models.generate_content(
        model="gemini-2.0-flash",  # Ensure this is the correct model name
        contents=[llm_prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=1000,
            temperature=0.1
        )
    )

    return response.text


# Define color map
color_map = {
    0: [70, 70, 70],     # Background - dark gray
    1: [14, 135, 204],   # Water - blue
    2: [51, 51, 0],      # Forest - dark green
    3: [9, 143, 150],    # Vehicles - teal
    4: [128, 64, 128],   # Roads - purple
    5: [225, 22, 96],    # People - pink
    6: [102, 102, 156],  # Urban areas - gray-blue
    7: [130, 76, 0],     # Barren land - brown
    8: [0, 102, 0],      # Vegetation - green
    9: [107, 142, 35]    # Grassland - olive
}

# Define class names (adjust according to your model)
class_names = (
    "Background",
    "Water",
    "Forest",
    "Vehicels",
    "Paved area",
    "People",
    "Roof",
    "Drit",
    "Grassland",
    "Vegetation"
)


# Streamlit UI
st.title("🚁 Drone Image Based Segmentation System 📸")

# Upload file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# User input for image capture height
image_height = st.number_input("Enter Image Capture Height (m)", min_value=5.0, max_value=100.0, value=80.0, step=0.5)

model = load_model("deeplabv3plus_model.keras", compile=False)

default_image_path = "80m/swiss_IMG_8720.jpeg"

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
else:
    # Load the default image
    image = cv2.imread(default_image_path)

st.subheader("⚠️ To use the default image, click the button below ⚠️")

if image is not None and (uploaded_file is not None or default_image_path):
    #file_bytes = uploaded_file.read()
    #if uploaded_file.type.startswith("image") or i:
        if st.button('Run Segmentation'):
            with st.spinner('Processing image and generating segmentation...'):
                #image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                # Convert to RGB (OpenCV loads as BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"Original image shape: {image.shape}")
                image = cv2.resize(image, (3000, 2000))
                print(f"Resized image shape: {image.shape}")

                cap_height = image_height

                if(cap_height <= 17.5):
                    # Adjust the image for the model
                    image = cv2.resize(image, (512, 512))
                    preprocess_image = np.expand_dims(image / 255.0, axis=0)
                    # Make predictions
                    result_labels = model.predict(preprocess_image)[0]
                    result_labels = np.argmax(result_labels, axis=-1)
                    result_rgb = visualize_segmented_image(result_labels, color_map)
                else:
                    # Perform height-adaptive sliding window segmentation
                    result_rgb, result_labels = height_adaptive_sliding_window(
                        image=image,
                        model=model,
                        color_map=color_map,
                        training_height=17.5,  # Height at which training images were captured
                        capture_height=cap_height,     # Height at which this image was captured
                        base_size=512,         # Size model was trained on
                        overlap_ratio=0.25,    # 25% overlap between windows
                        batch_size=4,          # Process 4 patches at once
                        verbose=True
                    )
                    image = cv2.resize(image, (512,512))
                    result_rgb = cv2.resize(result_rgb, (512,512))

                #print(f"Result labels shape: {result_labels.shape}")
                #result_labels = np.argmax(result_labels, axis=-1)
                #print(f"Result labels unique: {np.unique(result_labels)}")

                col1, col2 = st.columns([5, 5]) 
                with col1:
                    st.subheader("Original Image")
                    st.image(image, channels="RGB", caption="Original Image", use_container_width=True)
                with col2:
                    st.subheader("Segmented Image")
                    st.image(result_rgb, channels="RGB", caption="Segmented Image", use_container_width=True)

                # Calculate real-world area for each class
                segmented_pixels = np.count_nonzero(result_labels)
                total_area = calculate_real_world_area(segmented_pixels, image_height)
                
                class_areas = {}
                for class_id, class_name in enumerate(class_names):
                    pixel_count = np.count_nonzero(result_labels == class_id)
                    if pixel_count > 0:
                        class_areas[class_name] = calculate_real_world_area(pixel_count, image_height)

                col1, col2 = st.columns([5, 5]) 
                with col1:
                    # Display area calculations
                    st.subheader("🛰 Real-World Area Estimation")
                    st.write(f"Total Segmented Area: **{total_area:.2f} square meters**")
                    for class_name, area in class_areas.items():
                        st.write(f"- {class_name}: **{area:.2f} m²**")
                with col2:
                    st.subheader("💻 Class Distribution")
                    class_distribution_text = compute_class_metrics(result_labels, class_names)
                    
                st.subheader("🛰 LLM Drone Image Segmentation Analysis")
                with st.spinner('LLM Generating Response...'):
                    response = llm_analysis(class_distribution_text, image_height)
                st.write(response)

    
        