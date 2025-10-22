# trace_with_aruco.py
# Live pose from webcam + ArUco markers defined in tags.json
# Saves: trace.csv (x_mm,y_mm,theta_deg,timestamp) and trace.svg
import json, time, math, argparse
from pathlib import Path
import cv2
import numpy as np
from picamera2 import Picamera2
import os
from datetime import datetime

def load_tags_json(path):
    data = json.loads(Path(path).read_text())
    assert data["dict"] == "DICT_6X6_250"
    tag_size_mm = float(data["tag_size_mm"])
    tags_map = {int(k): v for k, v in data["tags"].items()}
    return tag_size_mm, tags_map

def build_object_points(tag_id, tag_size_mm, tag_map):
    # Return 3D corners (Z=0) of the tag in world coords, order must match detectMarkers corners
    # OpenCV aruco corners order: top-left, top-right, bottom-right, bottom-left
    # We stored tag map as center positions? No—above we stored top-left of each cell, *but*
    # for solvePnP we want the *tag corners in world frame*. Our generator places the marker’s
    # top-left at (x_mm, y_mm). So corners are:
    x0 = tag_map[tag_id]["x_mm"]
    y0 = tag_map[tag_id]["y_mm"]
    s = tag_size_mm
    corners = np.array([
        [x0,     y0,     0.0],
        [x0+s,   y0,     0.0],
        [x0+s,   y0+s,   0.0],
        [x0,     y0+s,   0.0],
    ], dtype=np.float32)
    return corners

def estimate_pose_from_tags(corners_list, ids, cam_mtx, dist, tag_size_mm, tag_map):
    # Stack 2D-3D correspondences from all visible tags
    obj_pts = []
    img_pts = []
    for c, tid in zip(corners_list, ids.flatten()):
        if tid not in tag_map: 
            continue
        obj = build_object_points(tid, tag_size_mm, tag_map)
        img = c.reshape(-1, 2).astype(np.float32)
        obj_pts.append(obj)
        img_pts.append(img)
    if len(obj_pts) == 0: 
        print(f"    No valid markers found in tag_map")
        return None, None
    
    obj_pts = np.concatenate(obj_pts, axis=0)
    img_pts = np.concatenate(img_pts, axis=0)
    print(f"    Using {len(obj_pts)} 3D-2D correspondences for pose estimation")
    
    # Try different solvePnP methods (avoid P3P due to ambiguity)
    methods_to_try = [
        (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
        (cv2.SOLVEPNP_EPNP, "EPNP"),
        (cv2.SOLVEPNP_DLS, "DLS"),
        (cv2.SOLVEPNP_UPNP, "UPNP")
    ]
    
    for method, name in methods_to_try:
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, cam_mtx, dist,
                                                         flags=method,
                                                         reprojectionError=5.0, confidence=0.95)
            if ok:
                inlier_count = len(inliers) if inliers is not None else 0
                print(f"    solvePnPRansac succeeded with {name} method, {inlier_count} inliers")
                
                # Check if we have enough inliers for stable pose estimation
                if inlier_count < 4:
                    print(f"    Warning: Only {inlier_count} inliers, pose might be unstable")
                
                return rvec, tvec
        except:
            continue
    
    print(f"    All solvePnP methods failed")
    return None, None

def rvec_tvec_to_xy_theta(rvec, tvec):
    # solvePnP returns the transform from world->camera
    # tvec is the translation from world origin to camera
    # For a stationary board, this should be relatively constant
    
    # Get the camera position in world coordinates
    x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
    
    # Get the rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Calculate yaw angle from the rotation matrix
    yaw = math.atan2(R[1,0], R[0,0])
    
    print(f"    Raw tvec: [{float(tvec[0]):.1f}, {float(tvec[1]):.1f}, {float(tvec[2]):.1f}]")
    print(f"    Raw rvec: [{float(rvec[0]):.1f}, {float(rvec[1]):.1f}, {float(rvec[2]):.1f}]")
    
    return x, y, yaw

def draw_axes(frame, cam_mtx, dist, rvec, tvec, scale=60.0):
    axis = np.float32([[0,0,0],[scale,0,0],[0,scale,0],[0,0,scale]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_mtx, dist)
    imgpts = imgpts.reshape(-1,2).astype(int)
    o,xp,yp,zp = imgpts
    cv2.line(frame, o, xp, 2, 2)
    cv2.line(frame, o, yp, 2, 2)
    cv2.line(frame, o, zp, 2, 2)

def save_svg_csv(samples, out_dir="out"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # CSV
    csvp = out/"trace.csv"
    with csvp.open("w") as f:
        f.write("t_sec,x_mm,y_mm,theta_deg\n")
        for t,x,y,th in samples:
            f.write(f"{t:.3f},{x:.3f},{y:.3f},{math.degrees(th):.3f}\n")
    # SVG (simple polyline)
    if not samples:
        return
    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    # Normalize to start at (0,0) and flip Y for SVG
    x0, y0 = xs[0], ys[0]
    pts = [(x - x0, -(y - y0)) for x, y in zip(xs, ys)]
    minx = min(p[0] for p in pts); miny = min(p[1] for p in pts)
    pts = [(p[0]-minx, p[1]-miny) for p in pts]
    w = max(p[0] for p in pts) + 10
    h = max(p[1] for p in pts) + 10
    svgp = out/"trace.svg"
    d = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}mm" height="{h:.1f}mm" viewBox="0 0 {w:.1f} {h:.1f}">
  <polyline fill="none" stroke="black" stroke-width="0.5" points="{d}"/>
</svg>'''
    svgp.write_text(svg)
    print(f"Saved {csvp} and {svgp}")

def load_camera_yaml(path):
    # YAML from OpenCV calibration: camera_matrix, dist_coeffs
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    cam_mtx = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    if cam_mtx is None or dist is None:
        raise ValueError("Could not read camera_matrix/dist_coeffs from YAML.")
    return cam_mtx, dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="out/tags.json", help="tags.json from generator")
    ap.add_argument("--calib", default="out/camera.yaml", help="OpenCV camera calibration YAML")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--minsamples", type=int, default=3)
    ap.add_argument("--debug-dir", default="debug/trace", help="Directory for debug frames")
    ap.add_argument("--save-debug", action="store_true", help="Save debug frames with pose visualization")
    ap.add_argument("--max-frames", type=int, default=30, help="Maximum number of frames to process")
    args = ap.parse_args()

    tag_size_mm, tag_map = load_tags_json(args.tags)
    cam_mtx, dist = load_camera_yaml(args.calib)

    # Create debug directory if saving debug frames
    if args.save_debug:
        debug_dir = args.debug_dir
        os.makedirs(debug_dir, exist_ok=True)

    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    det = cv2.aruco.ArucoDetector(dict, cv2.aruco.DetectorParameters())

    # Use Picamera2 instead of cv2.VideoCapture
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (640, 480)}))
    cam.start()

    samples = []
    t0 = time.time()
    frame_count = 0
    last_pose = None  # For pose filtering
    try:
        while frame_count < args.max_frames:
            frame = cam.capture_array()
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = det.detectMarkers(gray)
            
            if ids is not None and len(ids) >= args.minsamples:
                print(f"Frame {frame_count}: Detected {len(ids)} markers with IDs: {ids.flatten().tolist()}")
                
                # Check which markers are in our tag map
                valid_ids = [tid for tid in ids.flatten() if tid in tag_map]
                print(f"Frame {frame_count}: Valid markers in tag_map: {valid_ids}")
                
                rvec, tvec = estimate_pose_from_tags(corners, ids, cam_mtx, dist, tag_size_mm, tag_map)
                if rvec is not None:
                    x_mm, y_mm, yaw = rvec_tvec_to_xy_theta(rvec, tvec)
                    
                    # Simple pose filtering to avoid P3P ambiguity
                    if last_pose is not None:
                        dx = abs(x_mm - last_pose[0])
                        dy = abs(y_mm - last_pose[1])
                        dtheta = abs(yaw - last_pose[2])
                        
                        # If pose change is too large, it's likely the wrong solution
                        if dx > 50 or dy > 50 or dtheta > 30:
                            print(f"    Filtered out unstable pose: dx={dx:.1f}, dy={dy:.1f}, dtheta={dtheta:.1f}")
                            continue
                    
                    last_pose = (x_mm, y_mm, yaw)
                    samples.append((time.time()-t0, x_mm, y_mm, yaw))
                    
                    # Save debug frame if requested
                    if args.save_debug:
                        debug_frame = frame.copy()
                        cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)
                        draw_axes(debug_frame, cam_mtx, dist, rvec, tvec, scale=tag_size_mm)
                        cv2.putText(debug_frame, f"x={x_mm:.1f}mm y={y_mm:.1f}mm th={math.degrees(yaw):.1f}deg",
                                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        debug_filename = f"{debug_dir}/trace_frame_{frame_count:03d}_{timestamp}.jpg"
                        cv2.imwrite(debug_filename, debug_frame)
                        print(f"Saved debug frame: {debug_filename}")
                    
                    print(f"Frame {frame_count}: x={x_mm:.1f}mm y={y_mm:.1f}mm th={math.degrees(yaw):.1f}deg")
                else:
                    print(f"Frame {frame_count}: No valid pose detected (pose estimation failed)")
            else:
                print(f"Frame {frame_count}: No markers detected (need {args.minsamples})")
                # Save a sample frame to see what the camera is seeing
                if frame_count == 1 and args.save_debug:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    debug_filename = f"{debug_dir}/sample_frame_{timestamp}.jpg"
                    cv2.imwrite(debug_filename, frame)
                    print(f"  Saved sample frame: {debug_filename}")
            
            # Control frame rate
            time.sleep(1.0 / args.fps)
            
    except KeyboardInterrupt:
        print("\nStopping trace collection...")
    finally:
        cam.stop()
        print(f"\nProcessed {frame_count} frames. Collected {len(samples)} pose samples.")
        save_svg_csv(samples)

if __name__ == "__main__":
    main()
