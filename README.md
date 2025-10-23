# ArUco Vision Tracker

A simple vision tracking system using ArUco markers and ChArUco boards for 2D pose estimation with Raspberry Pi Camera.

## What it does

This system tracks the position (x, y, theta) of a camera relative to a printed ChArUco calibration board using computer vision. It outputs tracking data as CSV and SVG files.

**Key components:**
- [`generate_charuco_board.py`](generate_charuco_board.py:1) - Generates printable ChArUco boards
- [`calibrate_camera_charuco.py`](calibrate_camera_charuco.py:1) - Calibrates camera using the ChArUco board
- [`trace_with_aruco.py`](trace_with_aruco.py:1) - Tracks camera pose in real-time and saves trajectory data

## Quick Start

1. Generate a ChArUco board:
```bash
uv run python generate_charuco_board.py
```

2. Print the board from `out/charuco_board_print.png`

3. Calibrate your camera:
```bash
uv run python calibrate_camera_charuco.py
```

4. Track camera movement:
```bash
uv run python trace_with_aruco.py
```

Output files are saved to `out/trace.csv` and `out/trace.svg`.