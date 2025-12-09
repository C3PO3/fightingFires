"""
person_detection_module.py
Real-time person detection and coordinate mapping
Integrates with your existing optimized pipeline
"""
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

class PersonDetector:
    def __init__(self, model_size='n', conf_threshold=0.5):
        """
        Initialize YOLO + DeepSORT for person detection and tracking
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium) - nano recommended for speed
            conf_threshold: Detection confidence threshold (0.0-1.0)
        """
        # Load YOLO model
        model_name = f'yolov8{model_size}.pt'
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        
        # Configure for person detection only (class 0 in COCO dataset)
        self.person_class_id = 0
        self.conf_threshold = conf_threshold
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,              # Frames to keep track after disappearance
            n_init=3,                # Frames needed to confirm track
            max_iou_distance=0.7,    # IOU threshold for matching
            embedder="mobilenet",    # Use lightweight embedder
            half=True,               # Use FP16 for speed
        )
        
        # Performance monitoring
        self.detect_times = []
        self.track_times = []
        
        # Camera calibration (you'll need to calibrate your actual camera)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.setup_camera_calibration()
        
        print(f"Person Detector initialized (Model: YOLOv8{model_size})")
    
    def setup_camera_calibration(self):
        """
        Setup camera calibration parameters
        TODO: Replace with actual calibration from your camera setup
        """
        # Example calibration matrix (replace with your actual values)
        focal_length = 800  # pixels
        cx, cy = 320, 240   # principal point (image center)
        
        self.camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients (k1, k2, p1, p2, k3)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
    
    def detect_and_track(self, frame):
        """
        Detect and track persons in frame
        
        Returns:
            tracked_persons: List of dicts with format:
                {
                    'track_id': int,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'center_pixel': (x, y),
                    'bottom_center': (x, y)  # feet position
                }
        """
        detect_start = time.time()
        
        # YOLO detection (person class only)
        results = self.model(
            frame,
            classes=[self.person_class_id],  # Only detect persons
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        detect_time = time.time() - detect_start
        self.detect_times.append(detect_time)
        
        # Prepare detections for DeepSORT
        detections = []
        boxes = results.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # DeepSORT format: ([x1, y1, w, h], confidence, class)
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, 'person'))
        
        # Update tracker
        track_start = time.time()
        tracks = self.tracker.update_tracks(detections, frame=frame)
        track_time = time.time() - track_start
        self.track_times.append(track_time)
        
        # Format output
        tracked_persons = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = ltrb
            
            # Calculate center and bottom center (feet position)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            bottom_center_x = center_x
            bottom_center_y = int(y2)  # Bottom of bounding box
            
            tracked_persons.append({
                'track_id': track_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': track.get_det_conf() if hasattr(track, 'get_det_conf') else 1.0,
                'center_pixel': (center_x, center_y),
                'bottom_center': (bottom_center_x, bottom_center_y)
            })
        
        return tracked_persons
    
    def pixel_to_world_coordinates(self, pixel_coords, depth=None, camera_height=2.0):
        """
        Convert pixel coordinates to world coordinates
        
        Args:
            pixel_coords: (x, y) in pixels
            depth: Distance from camera (meters) - from LiDAR if available
            camera_height: Height of camera above ground (meters)
        
        Returns:
            (x, y, z) in world coordinates (meters)
        """
        if depth is None:
            # Estimate depth using simple pinhole camera model
            # This is a placeholder - integrate with LiDAR for accurate depth
            depth = 5.0  # Default 5 meters
        
        x_pixel, y_pixel = pixel_coords
        
        # Undistort pixel coordinates
        pixel_array = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(
            pixel_array, 
            self.camera_matrix, 
            self.dist_coeffs,
            P=self.camera_matrix
        )[0][0]
        
        # Convert to normalized camera coordinates
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Calculate 3D position
        x_world = (undistorted[0] - cx) * depth / fx
        y_world = (undistorted[1] - cy) * depth / fy
        z_world = depth
        
        return (x_world, y_world, z_world)
    
    def map_to_lidar_coordinates(self, persons, lidar_transform=None):
        """
        Map detected persons to LiDAR coordinate system
        
        Args:
            persons: List of tracked persons from detect_and_track()
            lidar_transform: 4x4 transformation matrix from camera to LiDAR frame
        
        Returns:
            persons with added 'world_coords' field
        """
        if lidar_transform is None:
            # Identity transform (assumes camera and LiDAR aligned)
            lidar_transform = np.eye(4)
        
        for person in persons:
            # Use bottom center (feet) for ground position
            pixel_pos = person['bottom_center']
            
            # TODO: Get actual depth from LiDAR point cloud
            depth = 5.0  # Placeholder
            
            # Convert to world coordinates
            world_pos = self.pixel_to_world_coordinates(pixel_pos, depth)
            
            # Transform to LiDAR frame
            world_homogeneous = np.array([*world_pos, 1.0])
            lidar_pos = (lidar_transform @ world_homogeneous)[:3]
            
            person['world_coords'] = {
                'x': float(lidar_pos[0]),
                'y': float(lidar_pos[1]),
                'z': float(lidar_pos[2])
            }
        
        return persons
    
    def draw_detections(self, frame, persons):
        """
        Draw bounding boxes and tracking info on frame
        """
        display_frame = frame.copy()
        
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            track_id = person['track_id']
            conf = person['confidence']
            
            # Draw bounding box
            color = self._get_track_color(track_id)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            center = person['center_pixel']
            cv2.circle(display_frame, center, 5, color, -1)
            
            # Draw bottom center (feet)
            bottom = person['bottom_center']
            cv2.circle(display_frame, bottom, 5, (0, 255, 255), -1)
            
            # Draw label
            label = f"ID:{track_id} {conf:.2f}"
            if 'world_coords' in person:
                wc = person['world_coords']
                label += f" ({wc['x']:.1f}, {wc['y']:.1f})"
            
            # Text background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def _get_track_color(self, track_id):
        """Generate consistent color for each track ID"""
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def get_performance_stats(self):
        """Get average detection and tracking times"""
        if not self.detect_times:
            return None
        
        avg_detect = np.mean(self.detect_times[-30:]) if self.detect_times else 0
        avg_track = np.mean(self.track_times[-30:]) if self.track_times else 0
        total = avg_detect + avg_track
        fps = 1.0 / total if total > 0 else 0
        
        return {
            'detect_ms': avg_detect * 1000,
            'track_ms': avg_track * 1000,
            'total_ms': total * 1000,
            'fps': fps
        }


# Example integration with your existing Data_Collector class
class Data_Collector_With_Detection:
    """
    Modified version of your Data_Collector that includes person detection
    """
    def __init__(self):
        # ... your existing init code ...
        
        # Add person detector
        self.person_detector = PersonDetector(
            model_size='n',  # Use nano for max speed
            conf_threshold=0.5
        )
        
        self.enable_detection = True  # Toggle detection on/off
        self.detection_interval = 1   # Process every N frames (1 = every frame)
        self.frame_counter = 0
        
        # Store detected persons
        self.current_persons = []
    
    def _process_frame_with_detection(self, rgb_frame):
        """
        Modified frame processing with person detection
        """
        self.frame_counter += 1
        
        # Run detection at specified interval
        if self.enable_detection and (self.frame_counter % self.detection_interval == 0):
            # Detect and track persons
            persons = self.person_detector.detect_and_track(rgb_frame)
            
            # Map to world coordinates (integrate with your LiDAR here)
            persons = self.person_detector.map_to_lidar_coordinates(persons)
            
            self.current_persons = persons
            
            # Draw detections
            rgb_frame = self.person_detector.draw_detections(rgb_frame, persons)
            
            # Print detection info
            if persons:
                print(f"Detected {len(persons)} person(s):")
                for p in persons:
                    print(f"  ID {p['track_id']}: {p['world_coords']}")
        
        return rgb_frame


# Standalone test
if __name__ == "__main__":
    # Test with webcam
    detector = PersonDetector(model_size='n')
    cap = cv2.VideoCapture(0)
    
    print("Starting person detection test...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track
        persons = detector.detect_and_track(frame)
        
        # Map to coordinates
        persons = detector.map_to_lidar_coordinates(persons)
        
        # Draw results
        display = detector.draw_detections(frame, persons)
        
        # Show performance
        stats = detector.get_performance_stats()
        if stats:
            cv2.putText(display, f"FPS: {stats['fps']:.1f} ({stats['total_ms']:.1f}ms)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Person Detection', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()