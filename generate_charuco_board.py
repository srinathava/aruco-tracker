#!/usr/bin/env python3
"""
Generate a ChArUco board and corresponding tags.json file.
This creates a printable ChArUco board and the tags.json file for tracking.
"""
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Generate ChArUco board and tags.json")
    ap.add_argument("--squares-x", type=int, default=5, help="Number of squares in X direction")
    ap.add_argument("--squares-y", type=int, default=7, help="Number of squares in Y direction")
    ap.add_argument("--square-mm", type=float, default=33.0, help="Square size in mm")
    ap.add_argument("--marker-mm", type=float, default=22.0, help="ArUco marker size in mm")
    ap.add_argument("--output-dir", default="out", help="Output directory")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for the printed board")
    args = ap.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), 
                                   args.square_mm, args.marker_mm, aruco_dict)
    
    # Generate board image with proper aspect ratio
    # Calculate the expected board size in pixels at high resolution
    high_res_dpi = 600  # High resolution for generation
    pixels_per_mm_hr = high_res_dpi / 25.4
    board_width_px_hr = int(args.squares_x * args.square_mm * pixels_per_mm_hr)
    board_height_px_hr = int(args.squares_y * args.square_mm * pixels_per_mm_hr)
    
    board_img = board.generateImage((board_width_px_hr, board_height_px_hr))
    
    # Save board image
    board_path = output_dir / "charuco_board.png"
    cv2.imwrite(str(board_path), board_img)
    print(f"Generated ChArUco board: {board_path}")
    
    # Get the actual board dimensions from OpenCV
    # The board.generateImage() creates a properly sized image
    # We just need to scale it to the correct DPI for printing
    pixels_per_mm = args.dpi / 25.4
    
    # Calculate the actual board dimensions
    # ChArUco board size is determined by the square size and number of squares
    board_width_mm = args.squares_x * args.square_mm
    board_height_mm = args.squares_y * args.square_mm
    
    # Scale the high-res image to print size
    board_width_px = int(board_width_mm * pixels_per_mm)
    board_height_px = int(board_height_mm * pixels_per_mm)
    
    # Resize for printing (maintain aspect ratio)
    board_print = cv2.resize(board_img, (board_width_px, board_height_px), interpolation=cv2.INTER_CUBIC)
    board_print_path = output_dir / "charuco_board_print.png"
    cv2.imwrite(str(board_print_path), board_print)
    print(f"Generated printable board ({board_width_mm:.1f}x{board_height_mm:.1f}mm): {board_print_path}")
    
    # Generate tags.json with marker positions
    tags = {}
    tag_id = 0
    
    # Calculate marker positions (top-left corners)
    for row in range(args.squares_y - 1):  # -1 because markers are in squares, not on edges
        for col in range(args.squares_x - 1):
            x_mm = col * args.square_mm
            y_mm = row * args.square_mm
            
            tags[str(tag_id)] = {
                "x_mm": x_mm,
                "y_mm": y_mm
            }
            tag_id += 1

    # Create the JSON structure
    data = {
        "dict": "DICT_6X6_250",
        "tag_size_mm": args.marker_mm,
        "tags": tags
    }

    # Write tags.json
    tags_path = output_dir / "tags.json"
    with open(tags_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated tags.json: {tags_path}")
    print(f"Board dimensions: {board_width_mm:.1f} x {board_height_mm:.1f} mm")
    print(f"Number of markers: {len(tags)}")
    print(f"Marker size: {args.marker_mm}mm")
    print(f"Square size: {args.square_mm}mm")
    print(f"\nTo use:")
    print(f"1. Print the board: {board_print_path}")
    print(f"2. Calibrate camera: python calibrate_camera_charuco.py --squares-x {args.squares_x} --squares-y {args.squares_y} --square-mm {args.square_mm} --marker-mm {args.marker_mm}")
    print(f"3. Run tracking: python trace_with_aruco.py --tags {tags_path} --calib camera.yaml")

if __name__ == "__main__":
    main()
