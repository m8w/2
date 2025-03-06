import cv2
import numpy as np
from skimage import measure
import os

# Load the image
input_image = "~/Desktop/Photos-symbol graphs/IMG_3890.JPG"  # First image in the folder
input_image = os.path.expanduser(input_image)  # Expand the ~ to full path
output_dir = "isolated_scribbles"  # Directory to save results

print(f"Processing image: {input_image}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the image
img = cv2.imread(input_image)
if img is None:
    print(f"Error: Could not read image at {input_image}")
    exit(1)
    
print(f"Image loaded successfully. Size: {img.shape}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to separate scribbles from background
# You may need to adjust this threshold based on your image
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Save the binary image for debugging
cv2.imwrite(os.path.join(output_dir, "binary_debug.jpg"), binary)
print("Saved binary debug image")

# Find connected components (scribbles)
labels = measure.label(binary, connectivity=2)
props = measure.regionprops(labels)
print(f"Found {len(props)} regions in the image")

# Process each scribble
count = 0
for i, prop in enumerate(props):
    # Filter out very small regions (noise)
    if prop.area < 100:  # Adjust this threshold as needed
        continue
        
    count += 1
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
    
    if i < 5 or i % 10 == 0:  # Only print status for the first few and then occasionally
        print(f"Saved {output_path}")

print(f"Processed {len(props)} potential scribbles, saved {count} that met size criteria")
