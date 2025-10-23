# camera_calibrate_charuco.py
# Calibrate from a printed ChArUco board (DICT_6X6_250).
# Usage:
#   uv run python camera_calibrate_charuco.py --squares-x 5 --squares-y 7 --square-mm 30 --marker-mm 20 --cam 0
import argparse, cv2, numpy as np
from picamera2 import Picamera2
import time
import os
from datetime import datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--squares-x", type=int, default=5)     # inner chessboard squares (cols)
    ap.add_argument("--squares-y", type=int, default=7)     # inner chessboard squares (rows)
    ap.add_argument("--square-mm", type=float, default=33)  # square size
    ap.add_argument("--marker-mm", type=float, default=22)  # aruco marker size inside each square
    ap.add_argument("--frames", type=int, default=40)       # how many good frames to collect
    ap.add_argument("--outfile", default="out/camera.yaml")
    ap.add_argument("--debug-dir", default="debug/calibrate")  # directory for debug images
    args = ap.parse_args()

    aruco = cv2.aruco
    dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard((args.squares_x, args.squares_y),
                               args.square_mm, args.marker_mm, dict)
    detector = aruco.ArucoDetector(dict, aruco.DetectorParameters())

    # Create debug directory
    debug_dir = args.debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (800, 600)}))
    cam.start()

    all_corners, all_ids = [], []
    imsize = None
    frame_count = 0
    print("Move the board around: tilt/rotate, cover the frame, vary distance. Press 'c' to capture, 'q' to finish.")
    for i in range(100):
        frame = cam.capture_array()
        frame_count += 1
        print(f"Captured frame {frame_count}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Create a copy for debug visualization
        debug_frame = frame.copy()
        
        if ids is not None and len(ids) > 0:
            print("Detected markers")
            # Draw detected ArUco markers
            aruco.drawDetectedMarkers(debug_frame, corners, ids)
            
            aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
            ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret and ch_corners is not None and ch_ids is not None and len(ch_ids) > 6:
                # Draw ChArUco corners in green
                cv2.aruco.drawDetectedCornersCharuco(debug_frame, ch_corners, ch_ids, (0,255,0))
                
                # Save debug image with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
                debug_filename = f"{debug_dir}/frame_{frame_count:03d}_{timestamp}_markers.jpg"
                cv2.imwrite(debug_filename, debug_frame)
                print(f"  Saved debug image: {debug_filename}")
                
                # Add to calibration data
                all_corners.append(ch_corners)
                all_ids.append(ch_ids)
                imsize = gray.shape[::-1]
                print(f"  captured {len(all_ids)} calibration frames")
        else:
            # Save frame even when no markers detected (for comparison)
            if frame_count % 5 == 0:  # Save every 5th frame without markers
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                debug_filename = f"{debug_dir}/frame_{frame_count:03d}_{timestamp}_no_markers.jpg"
                cv2.imwrite(debug_filename, debug_frame)
                print(f"  Saved frame without markers: {debug_filename}")
        
        time.sleep(1)
        if len(all_ids) >= 8:
            break

    if len(all_ids) < 8:
        raise SystemExit("Not enough views captured. Try again with more angles/distances.")

    # Calibrate
    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, imsize, None, None)
    print(f"RMS reprojection error: {rms:.3f}")
    fs = cv2.FileStorage(args.outfile, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.release()
    print(f"Wrote {args.outfile}")
    print(f"Debug images saved in: {debug_dir}")
    print(f"Check the debug images to verify marker detection is working correctly.")

if __name__ == "__main__":
    main()

