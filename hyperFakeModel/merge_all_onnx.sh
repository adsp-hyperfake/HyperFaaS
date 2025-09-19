#!/usr/bin/env bash
# merge_and_replace_onnx.sh – merge ONNX external data into original file and delete .onnx_data

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <directory> <path_to_merge_onnx.py>"
    echo "Example: $0 /path/to/models ./merge_onnx.py"
    exit 1
fi

TARGET_DIR="$1"
MERGE_SCRIPT="$2"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: directory $TARGET_DIR does not exist"
    exit 1
fi

if [ ! -f "$MERGE_SCRIPT" ]; then
    echo "Error: merge script $MERGE_SCRIPT not found"
    exit 1
fi

find "$TARGET_DIR" -type f -name "*.onnx" | while read -r onnx_file; do
    echo "Processing: $onnx_file"

    tmp_file="${onnx_file}.tmp"

    # Merge into a temporary file
    python "$MERGE_SCRIPT" "$onnx_file" "$tmp_file"

    # Replace the original file with the merged one
    mv "$tmp_file" "$onnx_file"
    echo "Replaced original ONNX file with merged version."

    # Delete the .onnx_data file if it exists
    data_file="${onnx_file}_data"
    if [ -f "$data_file" ]; then
        rm -f "$data_file"
        echo "Deleted external data file: $data_file"
    fi
done

echo "All ONNX files processed and replaced with merged versions."
