import cv2
import numpy as np
from skimage import measure
import os

# Load the image
input_image = "your_page_with_scribbles.jpg"  # Change to your input file
output_dir = "isolated_scribbles"  # Directory to save results

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the image
img = cv2.imread(input_image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to separate scribbles from background
# You may need to adjust this threshold based on your image
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find connected components (scribbles)
labels = measure.label(binary, connectivity=2)
props = measure.regionprops(labels)

# Process each scribble
for i, prop in enumerate(props):
    # Filter out very small regions (noise)
    if prop.area < 100:  # Adjust this threshold as needed
        continue
        
    # Get bounding box
    minr, minc, maxr, maxc = prop.bbox
    
    # Add some padding
    padding = 10
    minr = max(0, minr - padding)
    minc = max(0, minc - padding)
    maxr = min(img.shape[0], maxr + padding)
    maxc = min(img.shape[1], maxc + padding)
    
    # Extract the scribble with its original colors
    scribble = img[minr:maxr, minc:maxc].copy()
    
    # Create a mask for this specific scribble
    mask = (labels[minr:maxr, minc:maxc] == prop.label).astype(np.uint8) * 255
    
    # Apply the mask to keep only the scribble pixels
    scribble_masked = cv2.bitwise_and(scribble, scribble, mask=mask)
    
    # Set background to white (or any color you prefer)
    white_bg = np.ones_like(scribble) * 255
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    
    # Combine scribble with background
    final_scribble = cv2.add(scribble_masked, background)
    
    # Save the isolated scribble
    output_path = os.path.join(output_dir, f"scribble_{i+1}.jpg")
    cv2.imwrite(output_path, final_scribble)
    
    print(f"Saved {output_path}")

print(f"Processed {len(props)} potential scribbles")