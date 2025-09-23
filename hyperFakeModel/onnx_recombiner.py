#!/usr/bin/env python3
"""
Created using ChatGPT 5

merge_onnx.py – merge ONNX external data (.onnx_data) into a single ONNX file.

Usage:
    python merge_onnx.py input_model.onnx output_model.onnx
"""

import argparse
import onnx
import sys
from pathlib import Path


def merge_onnx(input_path: str, output_path: str):
    """Load an ONNX model (with external data) and save it as a single file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        sys.exit(1)

    print(f"Loading ONNX model: {input_path}")
    model = onnx.load(str(input_path), load_external_data=True)

    print(f"Saving merged ONNX model to: {output_path}")
    onnx.save(model, str(output_path), save_as_external_data=False)
    print("Done. All external data embedded inside the new ONNX file.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge ONNX external data (.onnx_data) into a single ONNX file."
    )
    parser.add_argument("input", help="Path to input ONNX model (.onnx)")
    parser.add_argument("output", help="Path to output merged ONNX model (.onnx)")
    args = parser.parse_args()

    merge_onnx(args.input, args.output)


if __name__ == "__main__":
    main()
