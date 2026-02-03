#!/bin/bash

OUTPUT_FOLDER="output_jan5"
# Create output directory if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# Path to your checkpoint
# CHECKPOINT="../checkpoints/nov27_batch4_DualStreamCNN_empty_background_zed_mod_ee_yz_0.15_x_0.05/best.pth"
CHECKPOINT="../checkpoints/jan5_onebox/best.pth"

# Directory containing the sequence images
INPUT_DIR="./captures/sequence"

# Get all left images (they will be our reference for pairing)
for left_img in ${INPUT_DIR}/*_left.png; do
    # Extract the base name without _left.png
    base_name=$(basename "$left_img" _left.png)
    
    # Construct the right image path
    right_img="${INPUT_DIR}/${base_name}_right.png"
    
    # Extract sequence number for output naming
    seq_num=$(echo "$base_name" | grep -oP 'seq_\d+')
    
    # Output path
    output_path="${OUTPUT_FOLDER}/${seq_num}.jpg"
    
    # Check if right image exists
    if [ -f "$right_img" ]; then
        echo "Processing: $base_name"
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