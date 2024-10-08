#!/bin/bash

# define input and output folder
input_folder="/data/01s2_original/imgs/axial_origin"
output_folder="/outputs_axial_01s2"
model_path="/checkpoints/axial/checkpoint_epoch15.pth"

if [ ! -d "$output_folder" ]; then
    echo "Output folder does not exist, creating it: $output_folder"
    mkdir -p "$output_folder"
fi

for input_file in "$input_folder"/*.png; do
    filename=$(basename "$input_file")

    output_file="$output_folder/$filename"
    echo "filename: $output_file"

    echo "Processing: $input_file -> $output_file"
    python predict.py -m "$model_path" -i "$input_file" -o "$output_file" -s 1.0

    if [ $? -ne 0 ]; then
        echo "Error processing $input_file, stopping the script."
        exit 1
    fi
done

echo "All files processed."