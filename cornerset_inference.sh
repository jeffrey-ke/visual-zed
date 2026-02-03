#!/bin/bash

OUTPUT_FOLDER="output_cornerset"
# Create output directory if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# Path to your checkpoint
CHECKPOINT="../checkpoints/dec13_cornerset/best.pth"

# Base directory for input images
BASE_DIR="../datasets/cornerset/render/render0"
LEFT_DIR="${BASE_DIR}/gripper_left_rgb"
RIGHT_DIR="${BASE_DIR}/gripper_right_rgb"

# Get all left images (they will be our reference for pairing)
for left_img in ${LEFT_DIR}/rgb_*.png; do
    # Extract the base filename (e.g., rgb_0000.png)
    filename=$(basename "$left_img")
    
    # Construct the right image path
    right_img="${RIGHT_DIR}/${filename}"
    
    # Output path - save in single output folder
    output_path="${OUTPUT_FOLDER}/${filename}"
    
    # Check if right image exists
    if [ -f "$right_img" ]; then
        echo "Processing: $filename"
        python test_inference.py \
            --checkpoint "$CHECKPOINT" \
            --left "$left_img" \
            --right "$right_img" \
            -o "$output_path"
    else
        echo "Warning: Right image not found for $left_img"
    fi
done

echo "Processing complete! Results saved in ${OUTPUT_FOLDER}/"