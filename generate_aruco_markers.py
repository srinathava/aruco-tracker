#!/usr/bin/env python3
"""
Generate an ArUco marker grid and corresponding tags.json file.
This creates a printable ArUco marker grid for tracking.
"""
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO

def main():
    ap = argparse.ArgumentParser(description="Generate ArUco marker grid and tags.json")
    ap.add_argument("--rows", type=int, default=4, help="Number of rows in the grid")
    ap.add_argument("--cols", type=int, default=8, help="Number of columns in the grid")
    ap.add_argument("--marker-mm", type=float, default=20.0, help="ArUco marker size in mm")
    ap.add_argument("--spacing-mm", type=float, default=5.0, help="Spacing between markers in mm")
    ap.add_argument("--output-dir", default="out", help="Output directory")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for the printed grid")
    ap.add_argument("--page-size", choices=["letter", "a4"], default="letter", help="PDF page size")
    args = ap.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Calculate grid dimensions
    total_markers = args.rows * args.cols
    grid_width_mm = args.cols * args.marker_mm + (args.cols - 1) * args.spacing_mm
    grid_height_mm = args.rows * args.marker_mm + (args.rows - 1) * args.spacing_mm
    
    # Generate grid image at high resolution
    high_res_dpi = 600  # High resolution for generation
    pixels_per_mm_hr = high_res_dpi / 25.4
    grid_width_px_hr = int(grid_width_mm * pixels_per_mm_hr)
    grid_height_px_hr = int(grid_height_mm * pixels_per_mm_hr)
    marker_size_px_hr = int(args.marker_mm * pixels_per_mm_hr)
    spacing_px_hr = int(args.spacing_mm * pixels_per_mm_hr)
    
    # Create white background
    grid_img = np.ones((grid_height_px_hr, grid_width_px_hr), dtype=np.uint8) * 255
    
    # Generate and place markers
    marker_id = 0
    tags_data = []
    
    for row in range(args.rows):
        for col in range(args.cols):
            if marker_id >= 250:  # DICT_6X6_250 has 250 markers
                print(f"Warning: Reached maximum markers (250) for DICT_6X6_250")
                break
            
            # Generate marker
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px_hr)
            
            # Calculate position
            x_pos = col * (marker_size_px_hr + spacing_px_hr)
            y_pos = row * (marker_size_px_hr + spacing_px_hr)
            
            # Place marker on grid
            grid_img[y_pos:y_pos+marker_size_px_hr, x_pos:x_pos+marker_size_px_hr] = marker_img
            
            # Store marker info for tags.json
            tags_data.append({
                "id": marker_id,
                "row": row,
                "col": col,
                "size_mm": args.marker_mm
            })
            
            marker_id += 1
    
    # Save grid image
    grid_path = output_dir / "aruco_grid.png"
    cv2.imwrite(str(grid_path), grid_img)
    print(f"Generated ArUco grid: {grid_path}")
    
    # Create PDF with exact dimensions
    pdf_path = output_dir / "aruco_grid.pdf"
    
    # Select page size
    base_page_size = letter if args.page_size == "letter" else A4
    
    # Convert grid dimensions to points (1 mm = 2.83465 points)
    grid_width_pt = grid_width_mm * mm
    grid_height_pt = grid_height_mm * mm
    
    # Determine orientation based on grid aspect ratio
    # Use landscape if grid is wider than it is tall
    if grid_width_mm > grid_height_mm:
        page_size = (base_page_size[1], base_page_size[0])  # Swap to landscape
        orientation = "landscape"
    else:
        page_size = base_page_size
        orientation = "portrait"
    
    page_width, page_height = page_size
    
    # Add margins (10mm on each side)
    margin = 10 * mm
    available_width = page_width - 2 * margin
    available_height = page_height - 2 * margin
    
    # Check if grid fits on page
    if grid_width_pt > available_width or grid_height_pt > available_height:
        print(f"Warning: Grid ({grid_width_mm:.1f}x{grid_height_mm:.1f}mm) is larger than available page area")
        print(f"Available area: {available_width/mm:.1f}x{available_height/mm:.1f}mm")
        print(f"Consider reducing marker size or spacing")
    
    # Create PDF
    c = canvas.Canvas(str(pdf_path), pagesize=page_size)
    
    # Center the grid on the page with margins
    x_offset = (page_width - grid_width_pt) / 2
    y_offset = (page_height - grid_height_pt) / 2
    
    # Ensure minimum margin
    x_offset = max(x_offset, margin)
    y_offset = max(y_offset, margin)
    
    # Convert OpenCV image to PIL format for ReportLab
    # OpenCV uses BGR, convert to RGB
    grid_img_rgb = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2RGB)
    
    # Create BytesIO object to hold the image
    img_buffer = BytesIO()
    from PIL import Image
    pil_img = Image.fromarray(grid_img_rgb)
    pil_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Draw image on PDF with exact dimensions
    c.drawImage(ImageReader(img_buffer), x_offset, y_offset,
                width=grid_width_pt, height=grid_height_pt)
    
    # Add metadata text at bottom
    c.setFont("Helvetica", 8)
    text_y = 15
    metadata_text = f"ArUco Grid: {args.rows}x{args.cols} | Marker: {args.marker_mm}mm | Spacing: {args.spacing_mm}mm | Dictionary: DICT_6X6_250 | Orientation: {orientation}"
    c.drawString(30, text_y, metadata_text)
    
    c.save()
    print(f"Generated PDF ({grid_width_mm:.1f}x{grid_height_mm:.1f}mm, {orientation}): {pdf_path}")
    
    print(f"\nGrid dimensions: {grid_width_mm:.1f} x {grid_height_mm:.1f} mm")
    print(f"Marker size: {args.marker_mm}mm")
    print(f"Spacing: {args.spacing_mm}mm")
    print(f"Total markers: {len(tags_data)}")
    print(f"Page size: {args.page_size}")
    print(f"\nTo use:")
    print(f"1. Print the PDF at 100% scale (no scaling): {pdf_path}")
    print(f"\nIMPORTANT: When printing, ensure 'Actual Size' or '100%' is selected, NOT 'Fit to Page'")

if __name__ == "__main__":
    main()