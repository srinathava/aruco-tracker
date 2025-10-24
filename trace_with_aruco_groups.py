# trace_with_aruco_groups_v2.py
# Live pose from webcam + ArUco markers organized in groups with dynamic world coordinate transmission
# Groups: 0-7, 8-15, 16-23, etc. Each group has fixed relative positions but unknown absolute positions.
# Saves: trace.csv (x_mm,y_mm,theta_deg,timestamp) and trace.svg
import time, math, argparse, logging
from pathlib import Path
import cv2
import numpy as np
from picamera2 import Picamera2
import os
from datetime import datetime

# Configure logging - will be reconfigured in main() to write to file
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Group configuration
GROUP_ROWS = 1  # Number of rows in each group's grid layout
GROUP_COLS = 8  # Number of columns in each group's grid layout
GROUP_SIZE = GROUP_ROWS * GROUP_COLS  # Total markers per group

def get_group_id(marker_id):
    """Get the group ID for a marker (e.g., marker 5 -> group 0, marker 12 -> group 1)"""
    return marker_id // GROUP_SIZE

def get_group_leader(group_id):
    """Get the leader (smallest ID) of a group"""
    return group_id * GROUP_SIZE

def get_group_members(group_id):
    """Get all marker IDs in a group"""
    start = group_id * GROUP_SIZE
    return list(range(start, start + GROUP_SIZE))

def build_group_marker_map(group_id, square_mm, marker_mm):
    """
    Build a corner map for a single group of markers.
    Each group has GROUP_SIZE markers arranged in a GROUP_ROWS x GROUP_COLS grid.
    The leader (smallest ID) is at position (0, 0).
    """
    dict6x6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Define layout for each group based on GROUP_ROWS x GROUP_COLS grid
    # Positions are relative to the group leader at (0, 0)
    group_layout = []
    for row in range(GROUP_ROWS):
        for col in range(GROUP_COLS):
            x = col * square_mm
            y = row * square_mm
            group_layout.append((x, y))
    
    corner_map = {}
    group_members = get_group_members(group_id)
    
    for i, marker_id in enumerate(group_members):
        # Get the center position for this marker
        cx, cy = group_layout[i]
        
        # Calculate the four corners of the marker
        # Corners are ordered: top-left, top-right, bottom-right, bottom-left
        half_size = marker_mm / 2.0
        corners = np.array([
            [cx - half_size, cy - half_size, 0],  # top-left
            [cx + half_size, cy - half_size, 0],  # top-right
            [cx + half_size, cy + half_size, 0],  # bottom-right
            [cx - half_size, cy + half_size, 0],  # bottom-left
        ], dtype=np.float32)
        
        corner_map[marker_id] = corners
    
    return corner_map, dict6x6

def estimate_pose_from_tags(corners_list, ids, cam_mtx, dist, corner_map):
    """Estimate camera pose from detected markers using their 3D positions in corner_map"""
    # Stack 2D-3D correspondences from all visible tags
    obj_pts = []
    img_pts = []
    for c, tid in zip(corners_list, ids.flatten()):
        if tid not in corner_map:
            continue
        obj = corner_map[tid]
        img = c.reshape(-1, 2).astype(np.float32)
        obj_pts.append(obj)
        img_pts.append(img)
    
    if len(obj_pts) == 0:
        logging.debug(f"    No valid markers found in corner map")
        return None, None
    
    obj_pts = np.concatenate(obj_pts, axis=0)
    img_pts = np.concatenate(img_pts, axis=0)
    logging.debug(f"    Using {len(obj_pts)} 3D-2D correspondences for pose estimation")
    
    # Try different solvePnP methods
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
                logging.debug(f"    solvePnPRansac succeeded with {name} method, {inlier_count} inliers")
                
                if inlier_count < 4:
                    logging.debug(f"    Warning: Only {inlier_count} inliers, pose might be unstable")
                
                return rvec, tvec
        except:
            continue
    
    logging.debug(f"    All solvePnP methods failed")
    return None, None

def transform_local_to_world(local_corner_map, rvec_local_to_cam, tvec_local_to_cam, rvec_world_to_cam, tvec_world_to_cam):
    """
    Transform marker positions from local group coordinates to world coordinates.
    
    We have two transformations:
    1. local → camera: given by rvec_local_to_cam, tvec_local_to_cam (from solvePnP on new group)
    2. world → camera: given by rvec_world_to_cam, tvec_world_to_cam (from solvePnP on known group)
    
    We want: local → world
    
    The composition is:
    p_cam = R_local_to_cam @ p_local + t_local_to_cam
    p_cam = R_world_to_cam @ p_world + t_world_to_cam
    
    Therefore:
    R_world_to_cam @ p_world + t_world_to_cam = R_local_to_cam @ p_local + t_local_to_cam
    R_world_to_cam @ p_world = R_local_to_cam @ p_local + t_local_to_cam - t_world_to_cam
    p_world = R_world_to_cam^(-1) @ (R_local_to_cam @ p_local + t_local_to_cam - t_world_to_cam)
    
    Note: For rotation matrices, R^(-1) = R^T because rotation matrices are orthogonal.
    This is a fundamental property: R @ R^T = R^T @ R = I
    So we can use the transpose instead of computing the inverse, which is more efficient.
    """
    # Get rotation matrices
    R_local_to_cam, _ = cv2.Rodrigues(rvec_local_to_cam)
    R_world_to_cam, _ = cv2.Rodrigues(rvec_world_to_cam)
    # For rotation matrices: R^(-1) = R^T (orthogonal property)
    R_cam_to_world = R_world_to_cam.T
    
    # Reshape tvecs
    t_local_to_cam = tvec_local_to_cam.reshape(3, 1)
    t_world_to_cam = tvec_world_to_cam.reshape(3, 1)
    
    # Transform each marker's corners
    transformed_map = {}
    for marker_id, corners in local_corner_map.items():
        # corners is (4, 3) array
        # Transform: p_world = R_cam_to_world @ (R_local_to_cam @ p_local + t_local_to_cam - t_world_to_cam)
        p_local = corners.T  # (3, 4)
        p_cam = R_local_to_cam @ p_local + t_local_to_cam  # (3, 4)
        p_world = R_cam_to_world @ (p_cam - t_world_to_cam)  # (3, 4)
        transformed_map[marker_id] = p_world.T.astype(np.float32)  # (4, 3)
    
    return transformed_map

def rvec_tvec_to_xy_theta(rvec, tvec):
    """Convert camera pose to x, y, theta representation"""
    # solvePnP returns the pose of the object (world) in the camera frame.
    # Convert that into the camera pose expressed in the world frame so that
    # results are independent of which group we used for estimation.
    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_to_world = R_world_to_cam.T
    cam_pos_world = -R_cam_to_world @ tvec.reshape(3, 1)
    x, y, z = cam_pos_world.flatten().tolist()
    logging.debug(f"    Raw tvec: [{float(tvec[0]):.1f}, {float(tvec[1]):.1f}, {float(tvec[2]):.1f}]")
    logging.debug(f"    Raw rvec: [{float(rvec[0]):.1f}, {float(rvec[1]):.1f}, {float(rvec[2]):.1f}]")
    logging.debug(f"    Camera position (world): [{x:.1f}, {y:.1f}, {z:.1f}]")

    # Yaw of the camera in world frame
    yaw = math.atan2(R_cam_to_world[1, 0], R_cam_to_world[0, 0])

    return x, y, yaw

def draw_axes(frame, cam_mtx, dist, rvec, tvec, scale=60.0):
    """Draw coordinate axes on the frame"""
    axis = np.float32([[0,0,0],[scale,0,0],[0,scale,0],[0,0,scale]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_mtx, dist)
    imgpts = imgpts.reshape(-1,2).astype(int)
    o,xp,yp,zp = imgpts
    cv2.line(frame, o, xp, (0,0,255), 2)  # X-axis: red
    cv2.line(frame, o, yp, (0,255,0), 2)  # Y-axis: green
    cv2.line(frame, o, zp, (255,0,0), 2)  # Z-axis: blue

class GroupTracker:
    """Manages state for tracking multiple marker groups across frames"""
    
    def __init__(self, square_mm, marker_mm, cam_mtx, dist, min_samples=3):
        self.square_mm = square_mm
        self.marker_mm = marker_mm
        self.cam_mtx = cam_mtx
        self.dist = dist
        self.min_samples = min_samples
        
        # State tracking
        self.known_groups = {}  # group_id -> corner_map in world coordinates
        self.active_group = None  # Currently used group for pose estimation
    
    def build_local_corner_map(self, group_id):
        """Build corner map for a group in its local coordinate system"""
        return build_group_marker_map(group_id, self.square_mm, self.marker_mm)
    
    def process_frame(self, frame_count, corners, ids):
        """
        Process a single frame with detected markers.
        Returns: (x_mm, y_mm, yaw, debug_info) or (None, None, None, debug_info)
        """
        debug_info = {
            'detected_groups': {},
            'active_group': self.active_group,
            'known_groups': list(self.known_groups.keys())
        }
        
        if ids is None or len(ids) == 0:
            logging.debug(f"  No markers detected")
            return None, None, None, debug_info
        
        detected_ids = ids.flatten().tolist()
        logging.debug(f"  Detected {len(ids)} markers with IDs: {detected_ids}")
        
        # Group detected markers by group ID
        groups_detected = {}
        for marker_id in detected_ids:
            group_id = get_group_id(marker_id)
            if group_id not in groups_detected:
                groups_detected[group_id] = []
            groups_detected[group_id].append(marker_id)
        
        debug_info['detected_groups'] = dict(groups_detected)
        logging.debug(f"  Detected groups: {dict(groups_detected)}")
        logging.debug(f"  Known groups: {list(self.known_groups.keys())}")
        logging.debug(f"  Active group: {self.active_group}")
        
        # Find which group to use for pose estimation
        group_to_use = self._select_active_group(groups_detected, frame_count, debug_info)
        
        if group_to_use is None:
            logging.debug(f"  No suitable group found for pose estimation")
            return None, None, None, debug_info
        
        # Get the corner map for this group and estimate pose
        corner_map = self.known_groups[group_to_use]
        rvec, tvec = estimate_pose_from_tags(corners, ids, self.cam_mtx, self.dist, corner_map)
        
        if rvec is None:
            logging.debug(f"  Pose estimation failed for group {group_to_use}")
            return None, None, None, debug_info
        
        # Successfully estimated pose
        x_mm, y_mm, yaw = rvec_tvec_to_xy_theta(rvec, tvec)
        self.active_group = group_to_use
        
        logging.debug(
            f"  Pose from group {group_to_use}: "
            f"x={x_mm:.1f}mm y={y_mm:.1f}mm th={math.degrees(yaw):.1f}deg"
        )
        debug_info['pose'] = (rvec, tvec)
        debug_info['group_used'] = group_to_use
        
        # Process all detected groups (transmit coordinates or sanity check)
        self._process_detected_groups(frame_count, groups_detected, corners, ids, rvec, tvec, debug_info)
        
        return x_mm, y_mm, yaw, debug_info
    
    def _select_active_group(self, groups_detected, frame_count, debug_info):
        """
        Select which group to use for pose estimation.
        Prioritizes known groups with the most detected markers.
        Once we have known groups, we refuse to initialize new groups to maintain
        a consistent world reference frame.
        """
        # Filter groups that have enough markers
        valid_groups = [(gid, len(markers))
                       for gid, markers in groups_detected.items()
                       if len(markers) >= self.min_samples]
        
        if not valid_groups:
            return None
        
        # Sort by: 1) known groups first (True > False), 2) most markers detected
        # Returning a tuple ensures proper lexicographic ordering
        def score_group(group_tuple):
            gid, marker_count = group_tuple
            is_known = gid in self.known_groups
            return (is_known, marker_count)
        
        valid_groups.sort(key=score_group, reverse=True)
        group_to_use, marker_count = valid_groups[0]
        
        # Check if we're trying to use an unknown group when we already have known groups
        if group_to_use not in self.known_groups and len(self.known_groups) > 0:
            logging.debug(
                f"  No known groups visible, skipping frame to maintain world reference"
            )
            return None
        
        # Log what we're doing
        if group_to_use in self.known_groups:
            if group_to_use == self.active_group:
                logging.debug(
                    f"  Continuing with active group {group_to_use} ({marker_count} markers)"
                )
            else:
                logging.debug(
                    f"  Switching to known group {group_to_use} ({marker_count} markers)"
                )
        else:
            # Only initialize new groups if we have no known groups yet
            logging.debug(
                f"  Initializing with new group {group_to_use} ({marker_count} markers)"
            )
            # Build corner map for this group with leader at (0, 0)
            corner_map, _ = self.build_local_corner_map(group_to_use)
            self.known_groups[group_to_use] = corner_map
        
        return group_to_use
    
    def _process_detected_groups(self, frame_count, groups_detected, corners, ids, rvec, tvec, debug_info):
        """Process all detected groups: transmit coordinates to new groups or sanity check known groups"""
        for detected_group_id in groups_detected:
            if len(groups_detected[detected_group_id]) < self.min_samples:
                logging.debug(f"    Failed to solve pose for group {detected_group_id}")
                continue
            
            # Build local corner map for this group
            logging.debug(f"Processing group {detected_group_id}")
            local_corner_map, _ = self.build_local_corner_map(detected_group_id)
            
            # Get this group's pose in camera frame
            rvec_detected, tvec_detected = estimate_pose_from_tags(
                corners, ids, self.cam_mtx, self.dist, local_corner_map
            )

            if rvec_detected is None:
                logging.debug(f"    Could not estimate pose for group {detected_group_id}")
                continue
            
            # Transform local coordinates to world coordinates
            world_corner_map = transform_local_to_world(
                local_corner_map, rvec_detected, tvec_detected, rvec, tvec
            )
            
            if detected_group_id in self.known_groups:
                # Sanity check for known groups
                self._sanity_check_group(
                    frame_count, detected_group_id, world_corner_map, debug_info
                )
            else:
                # Transmit world coordinates to new group
                self._transmit_to_new_group(
                    frame_count, detected_group_id, world_corner_map, debug_info
                )
    
    def _sanity_check_group(self, frame_count, group_id, world_corner_map, debug_info):
        """Verify transformation correctness for a known group"""
        logging.debug(f"  SANITY CHECK for known group {group_id}")
        
        # Compare transformed coordinates with stored world coordinates
        stored_map = self.known_groups[group_id]
        max_diff = 0.0
        for mid in world_corner_map.keys():
            if mid in stored_map:
                stored_center = stored_map[mid].mean(axis=0)
                transformed_center = world_corner_map[mid].mean(axis=0)
                diff = np.linalg.norm(stored_center - transformed_center)
                max_diff = max(max_diff, diff)                
        
        if max_diff > 10.0:
            logging.warning(
                f"      WARNING: Large difference detected ({max_diff:.2f}mm) - "
                "transformation may be incorrect!"
            )
        else:
            logging.debug(f"      Max difference: {max_diff:.2f}mm - transformation looks good")
    
    def _transmit_to_new_group(self, frame_count, group_id, world_corner_map, debug_info):
        """Transmit world coordinates to a newly detected group"""
        logging.debug(
            f"  Transmitting world coordinates to new group {group_id}"
        )
        
        # Store the new group
        self.known_groups[group_id] = world_corner_map
        logging.debug(f"  Group {group_id} added to known groups")
        
        # Print diagnostics
        logging.debug(f"    World coordinates for group {group_id}:")
        for marker_id in sorted(world_corner_map.keys()):
            corners_wc = world_corner_map[marker_id]
            center = corners_wc.mean(axis=0)
            logging.debug(
                f"      Marker {marker_id}: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
            )

def save_svg_csv(samples, out_dir="out"):
    """Save trace data to CSV and SVG files"""
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
    logging.info(f"Saved {csvp} and {svgp}")

def load_camera_yaml(path):
    """Load camera calibration from YAML file"""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    cam_mtx = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    if cam_mtx is None or dist is None:
        raise ValueError("Could not read camera_matrix/dist_coeffs from YAML.")
    return cam_mtx, dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="out/camera.yaml", help="OpenCV camera calibration YAML")
    ap.add_argument("--square-mm", type=float, default=25.0, help="Distance between markers in millimeters")
    ap.add_argument("--marker-mm", type=float, default=20.0, help="ArUco marker size in millimeters")
    ap.add_argument("--fps", type=float, default=1)
    ap.add_argument("--minsamples", type=int, default=3, help="Minimum markers needed from a group")
    ap.add_argument("--debug-dir", default="debug/trace_groups", help="Directory for debug frames")
    ap.add_argument("--raw-dir", default="debug/trace_groups_raw", help="Directory for raw unmodified frames")
    ap.add_argument("--save-debug", action="store_true", help="Save debug frames with pose visualization", default=True)
    ap.add_argument("--save-raw", action="store_true", help="Save raw unmodified frames", default=True)
    ap.add_argument("--max-frames", type=int, default=30, help="Maximum number of frames to process")
    ap.add_argument("--replay-dir", default=None, help="Directory containing JPEGs from a previous run (disables camera capture)")
    args = ap.parse_args()

    # Configure logging to write to file in debug directory
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    log_file = debug_dir / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Reconfigure logging to write to both console and file
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Override the initial configuration
    )
    logging.info(f"Logging to: {log_file}")

    cam_mtx, dist = load_camera_yaml(args.calib)

    # Create debug directories if saving frames
    if args.save_debug:
        debug_dir = args.debug_dir
        os.makedirs(debug_dir, exist_ok=True)
    
    if args.save_raw:
        raw_dir = args.raw_dir
        os.makedirs(raw_dir, exist_ok=True)

    # Get ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    det = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Determine if we're replaying from files or capturing live
    replay_mode = args.replay_dir is not None
    cam = None
    replay_files = []
    
    if replay_mode:
        # Load list of JPEG files from replay directory
        replay_path = Path(args.replay_dir)
        if not replay_path.exists():
            raise ValueError(f"Replay directory does not exist: {args.replay_dir}")
        
        replay_files = sorted(replay_path.glob("*.jpg")) + sorted(replay_path.glob("*.jpeg"))
        if not replay_files:
            raise ValueError(f"No JPEG files found in replay directory: {args.replay_dir}")
        
        logging.info(f"Replay mode: Found {len(replay_files)} JPEG files in {args.replay_dir}")
        logging.info(f"Will process up to {min(len(replay_files), args.max_frames)} frames")
    else:
        # Use Picamera2 for live capture
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (800, 600)}))
        cam.set_controls({"ExposureTime": 1000, "AnalogueGain": 2.0})
        cam.start()
        logging.info("Live capture mode: Using Picamera2")

    samples = []
    t0 = time.time()
    frame_count = 0
    
    # Initialize group tracker
    tracker = GroupTracker(args.square_mm, args.marker_mm, cam_mtx, dist, args.minsamples)
    
    try:
        max_frames_to_process = min(args.max_frames, len(replay_files)) if replay_mode else args.max_frames
        
        while frame_count < max_frames_to_process:
            logging.debug(f"\n{'='*60}")
            logging.debug(f"Frame {frame_count+1}")
            
            if replay_mode:
                # Load frame from file
                frame_file = replay_files[frame_count]
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logging.warning(f"  Failed to load {frame_file}, skipping...")
                    frame_count += 1
                    continue
            else:
                # Capture frame from camera
                frame = cam.capture_array()
            
            frame_count += 1
            
            # Save raw unmodified frame if requested (only in live capture mode)
            if args.save_raw and not replay_mode:
                raw_filename = f"{raw_dir}/frame_{frame_count:04d}.jpg"
                cv2.imwrite(raw_filename, frame)
                logging.debug(f"  Saved raw frame: {raw_filename}")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = det.detectMarkers(gray)

            # Save debug frame if requested
            if args.save_debug:
                debug_frame = frame.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)
            
            # Process frame using tracker
            x_mm, y_mm, yaw, debug_info = tracker.process_frame(frame_count, corners, ids)
            
            # If we got a valid pose, record it and update debug frame
            if x_mm is not None:
                samples.append((time.time()-t0, x_mm, y_mm, yaw))
                
                if args.save_debug and 'pose' in debug_info:
                    rvec, tvec = debug_info['pose']
                    draw_axes(debug_frame, cam_mtx, dist, rvec, tvec, scale=args.marker_mm)
                    cv2.putText(debug_frame, 
                                f"Group {debug_info['group_used']}: x={x_mm:.1f}mm y={y_mm:.1f}mm th={math.degrees(yaw):.1f}deg",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(debug_frame, 
                                f"Known groups: {debug_info['known_groups']}",
                                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            if args.save_debug:
                debug_filename = f"{debug_dir}/frame_{frame_count:04d}.jpg"
                cv2.imwrite(debug_filename, debug_frame)
                logging.debug(f"  Saved debug frame: {debug_filename}")
            
            # Control frame rate
            if not replay_mode:
                time.sleep(max(0.0, 1.0 / args.fps))
            
    except KeyboardInterrupt:
        logging.info("\nStopping trace collection...")
    finally:
        if cam is not None:
            cam.stop()
        logging.info(f"\nProcessed {frame_count} frames. Collected {len(samples)} pose samples.")
        logging.info(f"Total groups discovered: {len(tracker.known_groups)}")
        save_svg_csv(samples)

if __name__ == "__main__":
    main()