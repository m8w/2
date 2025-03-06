#!/usr/bin/env python3
"""
scribble_to_video.py - Convert graph paper scribbles to video frames

This script takes an image of graph paper with scribbles and creates a video
where each scribble becomes a frame in the video. The script automatically
detects the grid lines of the graph paper and uses them to segment the image
into individual scribbles based on grid cell positions.

Author: AI Assistant
"""

import argparse
import os
import sys
import cv2
import numpy as np
from collections import defaultdict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert graph paper scribbles to video frames.")
    parser.add_argument("input_image", help="Path to the input image file with scribbles on graph paper.")
    parser.add_argument("output_video", help="Path to the output video file.")
    parser.add_argument("--frame-duration", type=float, default=0.5,
                        help="Duration of each frame in seconds. Default: 0.5")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for the output video. Default: 30")
    parser.add_argument("--min-scribble-size", type=int, default=20,
                        help="Minimum size in pixels to consider as a scribble. Default: 20")
    parser.add_argument("--padding", type=int, default=10,
                        help="Padding around each scribble in pixels. Default: 10")
    parser.add_argument("--output-width", type=int, default=640,
                        help="Width of the output video. Default: 640")
    parser.add_argument("--output-height", type=int, default=480,
                        help="Height of the output video. Default: 480")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode to save intermediate processing images.")
    
    return parser.parse_args()


def detect_grid(image, debug=False, debug_dir=None):
    """
    Detect grid lines on graph paper.
    
    Args:
        image: Input image (grayscale)
        debug: Whether to save debug images
        debug_dir: Directory to save debug images
        
    Returns:
        tuple: (horizontal_lines, vertical_lines) where each is a list of line positions
    """
    # Make a copy for visualization
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if debug else None
    
    # Apply morphological operations to enhance grid lines
    kernel_size = max(1, min(image.shape) // 100)  # Adaptive kernel size based on image dimensions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Use morphological operations to enhance grid lines
    morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    # Edge detection
    edges = cv2.Canny(morph, 50, 150, apertureSize=3)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "1_edges.png"), edges)
    
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=min(image.shape) // 4, maxLineGap=20)
    
    if lines is None:
        print("No grid lines detected. The image might not contain a clear grid pattern.")
        return [], []
    
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line angle to determine if it's horizontal or vertical
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Horizontal lines have angles close to 0 or 180
        if angle < 20 or angle > 160:
            horizontal_lines.append((y1 + y2) // 2)  # Use average y-coordinate
            if debug:
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Vertical lines have angles close to 90
        elif 70 < angle < 110:
            vertical_lines.append((x1 + x2) // 2)  # Use average x-coordinate
            if debug:
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # Sort and remove duplicates
    horizontal_lines = sorted(set(horizontal_lines))
    vertical_lines = sorted(set(vertical_lines))
    
    # Filter lines that are too close to each other (merge nearby lines)
    horizontal_lines = merge_nearby_lines(horizontal_lines)
    vertical_lines = merge_nearby_lines(vertical_lines)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "2_detected_lines.png"), vis_image)
    
    return horizontal_lines, vertical_lines


def merge_nearby_lines(lines, threshold=10):
    """
    Merge lines that are too close to each other.
    
    Args:
        lines: List of line positions
        threshold: Maximum distance between lines to be merged
        
    Returns:
        list: Merged lines
    """
    if not lines:
        return []
    
    merged_lines = [lines[0]]
    
    for line in lines[1:]:
        if line - merged_lines[-1] <= threshold:
            # If current line is close to the last merged line, update the merged line position
            merged_lines[-1] = (merged_lines[-1] + line) // 2
        else:
            # Otherwise, add the line as a new entry
            merged_lines.append(line)
    
    return merged_lines


def identify_grid_cells(image, horizontal_lines, vertical_lines):
    """
    Identify grid cells based on detected horizontal and vertical lines.
    
    Args:
        image: Input image
        horizontal_lines: List of horizontal line positions
        vertical_lines: List of vertical line positions
        
    Returns:
        list: List of cell coordinates [(x1, y1, x2, y2), ...]
    """
    cells = []
    
    # Make sure we have at least two lines in each direction to form cells
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        print("Not enough grid lines detected to form cells.")
        return cells
    
    # Add image boundaries if missing
    if horizontal_lines[0] > 10:
        horizontal_lines.insert(0, 0)
    if horizontal_lines[-1] < image.shape[0] - 10:
        horizontal_lines.append(image.shape[0])
    
    if vertical_lines[0] > 10:
        vertical_lines.insert(0, 0)
    if vertical_lines[-1] < image.shape[1] - 10:
        vertical_lines.append(image.shape[1])
    
    # Create grid cells from the intersections of horizontal and vertical lines
    for i in range(len(vertical_lines) - 1):
        for j in range(len(horizontal_lines) - 1):
            x1 = vertical_lines[i]
            y1 = horizontal_lines[j]
            x2 = vertical_lines[i + 1]
            y2 = horizontal_lines[j + 1]
            
            # Only consider cells that are reasonably sized
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                cells.append((x1, y1, x2, y2))
    
    return cells


def extract_scribbles(image, cells, min_scribble_size=20, padding=10, debug=False, debug_dir=None):
    """
    Extract scribbles from grid cells.
    
    Args:
        image: Input image
        cells: List of cell coordinates [(x1, y1, x2, y2), ...]
        min_scribble_size: Minimum size in pixels to consider as a scribble
        padding: Padding around each scribble in pixels
        debug: Whether to save debug images
        debug_dir: Directory to save debug images
        
    Returns:
        list: List of scribble images
    """
    # Create a binary version of the image for content detection
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "3_binary.png"), binary)
    
    # Initialize dictionary to hold scribbles for merging adjacent cells
    cell_groups = defaultdict(list)
    cell_contents = {}
    
    # Process each cell
    for idx, (x1, y1, x2, y2) in enumerate(cells):
        # Extract the cell region with additional padding
        cell_roi = binary[max(0, y1-padding):min(binary.shape[0], y2+padding), 
                          max(0, x1-padding):min(binary.shape[1], x2+padding)]
        
        # Skip cells that are too small
        if cell_roi.shape[0] <= 0 or cell_roi.shape[1] <= 0:
            continue
        
        # Check if the cell contains a scribble (non-white pixels)
        non_white_pixels = np.sum(cell_roi > 0)
        
        if non_white_pixels > min_scribble_size:
            # Store the cell and its content
            cell_contents[idx] = {
                'bbox': (x1, y1, x2, y2),
                'content': cell_roi.copy(),
                'pixel_count': non_white_pixels
            }
            
            # Initially, each cell is in its own group
            cell_groups[idx].append(idx)
    
    # Merge adjacent cells with content (to handle scribbles that span multiple cells)
    merged_groups = merge_adjacent_cells(cells, cell_contents)
    
    # Extract scribbles from merged cell groups
    scribbles = []
    
    vis_image = None
    if debug:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for group_idx, group_cells in enumerate(merged_groups):
        if not group_cells:
            continue
        
        # Find the bounding box that contains all cells in the group
        min_x = min(cell_contents[cell_idx]['bbox'][0] for cell_idx in group_cells)
        min_y = min(cell_contents[cell_idx]['bbox'][1] for cell_idx in group_cells)
        max_x = max(cell_contents[cell_idx]['bbox'][2] for cell_idx in group_cells)
        max_y = max(cell_contents[cell_idx]['bbox'][3] for cell_idx in group_cells)
        
        # Add padding to the bounding box
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(binary.shape[1], max_x + padding)
        max_y = min(binary.shape[0], max_y + padding)
        
        # Extract the scribble region
        scribble_roi = binary[min_y:max_y, min_x:max_x]
        
        # Skip if the roi is empty
        if scribble_roi.size == 0:
            continue
        
        # Add to the list of scribbles
        scribbles.append({
            'image': scribble_roi,
            'bbox': (min_x, min_y, max_x, max_y)
        })
        
        if debug:
            cv2.rectangle(vis_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            
    if debug and vis_image is not None:
        cv2.imwrite(os.path.join(debug_dir, "4_detected_scribbles.png"), vis_image)
        
        # Save individual scribbles
        os.makedirs(os.path.join(debug_dir, "scribbles"), exist_ok=True)
        for i, scribble in enumerate(scribbles):
            cv2.imwrite(os.path.join(debug_dir, f"scribbles/scribble_{i}.png"), scribble['image'])
    
    return scribbles


def merge_adjacent_cells(cells, cell_contents):
    """
    Merge adjacent cells that likely contain parts of the same scribble.
    
    Args:
        cells: List of cell coordinates
        cell_contents: Dictionary of cell contents
        
    Returns:
        list: List of merged cell groups
    """
    # Create a graph of adjacent cells
    adjacency = defaultdict(list)
    
    # Check each pair of cells for adjacency
    cell_indices = list(cell_contents.keys())
    for i in range(len(cell_indices)):
        for j in range(i + 1, len(cell_indices)):
            idx1 = cell_indices[i]
            idx2 = cell_indices[j]
            
            bbox1 = cell_contents[idx1]['bbox']
            bbox2 = cell_contents[idx2]['bbox']
            
            # Cells are adjacent if they share an edge
            if (abs(bbox1[0] - bbox2[2]) <= 5 or abs(bbox1[2] - bbox2[0]) <= 5) and \
               (bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1]):
                # Horizontal adjacency
                adjacency[idx1].append(idx2)
                adjacency[idx2].append(idx1)
                
            elif (abs(bbox1[1] - bbox2[3]) <= 5 or abs(bbox1[3] - bbox2[1]) <= 5) and \
                 (bbox1[0] < bbox2[2] and bbox1[2] > bbox2[0]):
                # Vertical adjacency
                adjacency[idx1].append(idx2)
                adjacency[idx2].append(idx1)
    
    # Use DFS to find connected components (groups of adjacent cells)
    visited = set()
    merged_groups = []
    
    for idx in cell_indices:
        if idx in visited:
            continue
            
        # Start a new group
        group = []
        stack = [idx]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
                
            visited.add(current)
            group.append(current)
            
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        merged_groups.append(group)
    
    return merged_groups


def create_video(scribbles, output_path, frame_duration, fps, output_width, output_height):
    """
    Create a video from scribble images.
    
    Args:
        scribbles: List of scribble images and bounding boxes
        output_path: Path to save the output video
        frame_duration: Duration of each frame in seconds
        fps: Frames per second for the output video
        output_width: Width of the output video
        output_height: Height of the output video
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not scribbles:
        print("No scribbles detected to create video.")
        return False
    
    try:
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return False
        
        # Calculate number of frames for each scribble based on frame duration
        frames_per_scribble = int(frame_duration * fps)
        
        # For each scribble, create multiple frames
        for scribble in scribbles:
            # Get the scribble image
            scribble_img = scribble['image']
            
            # Create a blank canvas for the frame
            frame = np.ones((output_height, output_width), dtype=np.uint8) * 255
            
            # Calculate position to place the scribble in the center of the frame
            target_height = min(int(output_height * 0.8), scribble_img.shape[0])
            target_width = min(int(output_width * 0.8), scribble_img.shape[1])
            
            # Only resize if the scribble is too large
            if scribble_img.shape[0] > target_height or scribble_img.shape[1] > target_width:
                # Maintain aspect ratio
                aspect_ratio = scribble_img.shape[1] / scribble_img.shape[0]
                if target_height * aspect_ratio <= target_width:
                    target_width = int(target_height * aspect_ratio)
                else:
                    target_height = int(target_width / aspect_ratio)
                
                # Resize the scribble
                scribble_img = cv2.resize(scribble_img, (target_width, target_height), 
                                         interpolation=cv2.INTER_AREA)
            
            # Calculate position to center the scribble
            x_offset = (output_width - scribble_img.shape[1]) // 2
            y_offset = (output_height - scribble_img.shape[0]) // 2
            
            # Create region of interest in the frame
            roi = frame[y_offset:y_offset+scribble_img.shape[0], 
                        x_offset:x_offset+scribble_img.shape[1]]
            
            # Place the scribble on the frame (invert colors since scribble is in binary format)
            # The scribble is white on black, but we want black on white
            roi[scribble_img > 0] = 0
            
            # Convert to BGR for video
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Write the frame multiple times based on the frame duration
            for _ in range(frames_per_scribble):
                video_writer.write(frame_bgr)
        
        # Release resources
        video_writer.release()
        print(f"Video successfully created at: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False


def process_image(input_path, output_path, args):
    """
    Process the input image and create the output video.
    
    Args:
        input_path: Path to the input image
        output_path: Path to the output video
        args: Command-line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create debug directory if needed
        debug_dir = None
        if args.debug:
            debug_dir = os.path.splitext(output_path)[0] + "_debug"
            os.makedirs(debug_dir, exist_ok=True)
        
        # Read the input image
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not read input image {input_path}")
            return False
        
        print(f"Processing image: {input_path} (size: {image.shape[1]}x{image.shape[0]})")
        
        # Detect grid lines
        horizontal_lines, vertical_lines = detect_grid(image, args.debug, debug_dir)
        
        if not horizontal_lines or not vertical_lines:
            print("Warning: Grid lines detection failed. The script may not work as expected.")
        
        # Identify grid cells
        cells = identify_grid_cells(image, horizontal_lines, vertical_lines)
        
        if not cells:
            print("No grid cells detected. Please check if the image contains visible grid lines.")
            return False
        
        # Extract scribbles from cells
        scribbles = extract_scribbles(image, cells, 
                                     min_scribble_size=args.min_scribble_size,
                                     padding=args.padding,
                                     debug=args.debug, 
                                     debug_dir=debug_dir)
        
        if not scribbles:
            print("No scribbles detected in the image.")
            return False
        
        print(f"Detected {len(scribbles)} scribbles")
        
        # Create the output video
        return create_video(scribbles, output_path, 
                           args.frame_duration, args.fps,
                           args.output_width, args.output_height)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False


def main():
    """Main function to run the script."""
    try:
        # Parse command-line arguments
        args = parse_args()
        
        # Check if input file exists
        if not os.path.isfile(args.input_image):
            print(f"Error: Input file {args.input_image} does not exist")
            return 1
        
        # Check if output directory exists, create if needed
        output_dir = os.path.dirname(args.output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process the image and create the video
        success = process_image(args.input_image, args.output_video, args)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
