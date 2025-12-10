"""
main_with_person_detection.py
Location: person_recognition/main_with_person_detection.py
Integrates person detection into your existing optimized pipeline
Press 'q' to quit, 'e' to toggle edges, 'p' to pause, 'd' to toggle detection
"""
import sys
import os

# Add current_optimization_tests directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
current_optimization_tests_path = os.path.join(parent_dir, 'current_optimization_tests')

# Verify the path exists
if not os.path.exists(current_optimization_tests_path):
    print(f"ERROR: Cannot find current_optimization_tests folder at: {current_optimization_tests_path}")
    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print("\nExpected folder structure:")
    print("  project_root/")
    print("  ├── current_optimization_tests/")
    print("  │   ├── mock_rgb_optimized.py")
    print("  │   └── mock_thermal_optimized.py")
    print("  └── person_recognition/")
    print("      ├── main_with_person_detection.py (this file)")
    print("      └── person_detection_module.py")
    sys.exit(1)

sys.path.insert(0, current_optimization_tests_path)

try:
    from mock_rgb_optimized import Rgb_Handler
    from mock_thermal_optimized import Thermal_Handler
    from person_detection_module import PersonDetector
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print(f"\nMake sure these files exist:")
    print(f"  - {os.path.join(current_optimization_tests_path, 'mock_rgb_optimized.py')}")
    print(f"  - {os.path.join(current_optimization_tests_path, 'mock_thermal_optimized.py')}")
    print(f"  - {os.path.join(current_dir, 'person_detection_module.py')}")
    sys.exit(1)
import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Event
import queue

def rgb_capture_process(frame_queue, stop_event, fps_cap=60):
    """Separate process for RGB frame capture"""
    handler = Rgb_Handler(2, fps_cap)
    print(f"RGB capture process started (PID: {os.getpid()})")
    
    try:
        while not stop_event.is_set():
            frame = handler.get_frame()
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        handler.release()
        print("RGB capture process stopped")

def thermal_capture_process(frame_queue, stop_event, fps_cap=60):
    """Separate process for Thermal frame capture"""
    handler = Thermal_Handler(0, fps_cap)
    print(f"Thermal capture process started (PID: {os.getpid()})")
    
    try:
        while not stop_event.is_set():
            frame = handler.get_frame()
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        handler.release()
        print("Thermal capture process stopped")

class Data_Collector():
    def __init__(self):
        # State
        self.take_pictures = True
        self.display_mode = 'raw'  # 'raw' or 'edges'
        self.enable_detection = True  # Person detection toggle
        self.detection_interval = 1  # Detect every N frames (1=every frame, 2=every other, etc)
        
        # Image saving
        self.img_num = 0
        self.save_images = False
        self.usb_path_rgb = "./test_output/rgb_images"
        self.usb_path_thm = "./test_output/thm_images"
        
        if self.save_images:
            os.makedirs(self.usb_path_rgb, exist_ok=True)
            os.makedirs(self.usb_path_thm, exist_ok=True)
        
        # Performance monitoring
        self.frame_times = []
        self.last_fps_print = time.time()
        
        # Multiprocessing setup
        self.rgb_queue = Queue(maxsize=2)
        self.thermal_queue = Queue(maxsize=2)
        self.stop_event = Event()
        
        # Current frames
        self.current_rgb = None
        self.current_thermal = None
        
        # Pre-allocate buffers
        self.gray_buffer = None
        self.blur_buffer = None
        self.edge_buffer = None
        
        # Thermal colormap
        self.colormaps = [
            (cv2.COLORMAP_JET, "JET"),
            (cv2.COLORMAP_HOT, "HOT"),
            (cv2.COLORMAP_INFERNO, "INFERNO"),
            (cv2.COLORMAP_TURBO, "TURBO")
        ]
        self.current_colormap_idx = 0
        self.thermal_display_buffer = None
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        
        # FPS calculation
        self.fps_display = 0.0
        self.fps_update_counter = 0
        self.frame_counter = 0
        
        # Person detection
        print("Initializing person detector...")
        self.person_detector = PersonDetector(
            model_size='n',  # 'n' for nano (fastest), 's' for small (more accurate)
            conf_threshold=0.5
        )
        self.current_persons = []
        
        # Create OpenCV windows
        cv2.namedWindow('RGB Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Thermal Feed', cv2.WINDOW_NORMAL)
        
        cv2.moveWindow('RGB Feed', 0, 0)
        cv2.moveWindow('Thermal Feed', 650, 0)
        
        cv2.resizeWindow('RGB Feed', 640, 480)
        cv2.resizeWindow('Thermal Feed', 640, 480)
    
    def _get_latest_frames(self):
        """Get latest frames from queues (non-blocking)"""
        try:
            while not self.rgb_queue.empty():
                self.current_rgb = self.rgb_queue.get_nowait()
        except queue.Empty:
            pass
        
        try:
            while not self.thermal_queue.empty():
                self.current_thermal = self.thermal_queue.get_nowait()
        except queue.Empty:
            pass
    
    def _process_frame(self):
        """Process a single frame with person detection"""
        render_start = time.time()
        
        # Get latest frames
        self._get_latest_frames()
        
        if self.current_rgb is None or self.current_thermal is None:
            return
        
        rgb_frame = self.current_rgb.copy()  # Copy for processing
        thermal_frame = self.current_thermal
        
        # Initialize buffers on first frame
        if self.gray_buffer is None:
            h, w = rgb_frame.shape[:2]
            self.gray_buffer = np.empty((h, w), dtype=np.uint8)
            self.blur_buffer = np.empty((h, w), dtype=np.uint8)
            self.edge_buffer = np.empty((h, w), dtype=np.uint8)
        
        if self.thermal_display_buffer is None:
            h, w = thermal_frame.shape[:2]
            self.thermal_display_buffer = np.empty((h, w, 3), dtype=np.uint8)
        
        # Person detection (at specified interval)
        self.frame_counter += 1
        if self.enable_detection and (self.frame_counter % self.detection_interval == 0):
            # Detect and track persons
            detection_start = time.time()
            self.current_persons = self.person_detector.detect_and_track(rgb_frame)
            
            # Map to camera-relative coordinates
            # TODO: Replace None with actual LiDAR depth lookup function
            self.current_persons = self.person_detector.map_to_camera_relative_coordinates(
                self.current_persons,
                lidar_depth_map=None  # Will use default depth estimation
            )
            detection_time = (time.time() - detection_start) * 1000
            
            # Print detection results
            if self.current_persons and self.frame_counter % 30 == 0:  # Print every 30 frames
                print(f"\n--- Detected {len(self.current_persons)} person(s) ---")
                for p in self.current_persons:
                    coords = p.get('camera_relative_coords', {})
                    conf = p.get('confidence', 0.0)
                    if conf is None:
                        conf = 0.0
                    x = coords.get('x', 0)
                    y = coords.get('y', 0)
                    direction = "right" if x > 0 else "left"
                    print(f"  ID {p['track_id']}: "
                          f"{abs(x):.2f}m {direction}, {y:.2f}m forward | "
                          f"Conf: {conf:.2f}")
        
        # Draw detections on RGB frame
        if self.enable_detection and self.current_persons:
            rgb_frame = self.person_detector.draw_detections(rgb_frame, self.current_persons)
        
        # Process RGB based on display mode
        if self.display_mode == 'edges':
            cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY, dst=self.gray_buffer)
            cv2.GaussianBlur(self.gray_buffer, (3, 3), 0, dst=self.blur_buffer)
            cv2.Canny(self.blur_buffer, 50, 100, edges=self.edge_buffer)
            rgb_display = cv2.cvtColor(self.edge_buffer, cv2.COLOR_GRAY2BGR)
        else:
            rgb_display = rgb_frame
        
        # Convert thermal to color
        colormap, _ = self.colormaps[self.current_colormap_idx]
        cv2.applyColorMap(thermal_frame, colormap, dst=self.thermal_display_buffer)
        thermal_display = self.thermal_display_buffer
        
        # Update FPS calculation
        self.fps_update_counter += 1
        if self.fps_update_counter >= 5 and len(self.frame_times) > 0:
            avg_time = sum(self.frame_times[-5:]) / min(5, len(self.frame_times))
            self.fps_display = 1.0 / avg_time if avg_time > 0 else 0
            self.fps_update_counter = 0
        
        # Add overlays
        if self.fps_display > 0:
            mode_text = "EDGES" if self.display_mode == 'edges' else "RAW"
            detect_text = "DET:ON" if self.enable_detection else "DET:OFF"
            person_count = len(self.current_persons) if self.enable_detection else 0
            
            # RGB overlay
            cv2.putText(rgb_display, 
                       f"FPS: {self.fps_display:.1f} | {mode_text} | {detect_text} | P:{person_count}", 
                       (10, 30), self.font, self.font_scale, (0, 255, 0), self.font_thickness)
            
            # Thermal overlay
            cv2.putText(thermal_display, 
                       f"FPS: {self.fps_display:.1f} | Frame:{self.img_num}", 
                       (10, 30), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            # Detection performance stats
            if self.enable_detection:
                stats = self.person_detector.get_performance_stats()
                if stats:
                    cv2.putText(rgb_display,
                               f"Det: {stats['detect_ms']:.1f}ms | Track: {stats['track_ms']:.1f}ms",
                               (10, 60), self.font, 0.5, (0, 255, 255), 1)
        
        # Display frames
        cv2.imshow('RGB Feed', rgb_display)
        cv2.imshow('Thermal Feed', thermal_display)
        
        # Save images if enabled
        if self.save_images and self.take_pictures:
            filename_rgb = f"rgb{self.img_num}.jpg"
            filename_thm = f"thm{self.img_num}.jpg"
            save_path_rgb = os.path.join(self.usb_path_rgb, filename_rgb)
            save_path_thm = os.path.join(self.usb_path_thm, filename_thm)
            cv2.imwrite(save_path_rgb, rgb_frame)
            cv2.imwrite(save_path_thm, thermal_frame)
        
        if self.take_pictures:
            self.img_num += 1
        
        # Performance monitoring
        render_time = time.time() - render_start
        self.frame_times.append(render_time)
        
        if len(self.frame_times) > 30:
            self.frame_times = self.frame_times[-30:]
        
        # Print FPS every second
        if time.time() - self.last_fps_print >= 1.0:
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                mode_display = "EDGES" if self.display_mode == 'edges' else "RAW"
                detect_status = "ON" if self.enable_detection else "OFF"
                print(f"FPS: {fps:.2f} | Frame Time: {avg_frame_time*1000:.2f}ms | "
                      f"Frames: {self.img_num} | Mode: {mode_display} | Detection: {detect_status} | "
                      f"Persons: {len(self.current_persons)}")
            self.frame_times = []
            self.last_fps_print = time.time()
    
    def begin_data_stream(self):
        print("\n" + "="*70)
        print("CAMERA SYSTEM WITH PERSON DETECTION")
        print("="*70)
        print("Features: Edge Detection + Person Tracking + Coordinate Mapping")
        print("")
        print("Keyboard Controls:")
        print("  'q' or ESC - Quit")
        print("  'e' - Toggle between RAW and EDGE detection")
        print("  'd' - Toggle person detection ON/OFF")
        print("  'p' - Pause/Resume")
        print("  't' - Cycle thermal colormap")
        print("  '1-9' - Set detection interval (1=every frame, 2=every other, etc)")
        print("")
        print(f"Person Detection: {'ENABLED' if self.enable_detection else 'DISABLED'}")
        print(f"Detection Model: YOLOv8-nano + DeepSORT")
        print("="*70 + "\n")
        
        # Start capture processes
        rgb_process = Process(target=rgb_capture_process, 
                            args=(self.rgb_queue, self.stop_event, 60))
        thermal_process = Process(target=thermal_capture_process, 
                                args=(self.thermal_queue, self.stop_event, 60))
        
        rgb_process.start()
        thermal_process.start()
        
        try:
            while True:
                if self.take_pictures:
                    self._process_frame()
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("\nQuitting...")
                    break
                elif key == ord('e'):
                    self.display_mode = 'edges' if self.display_mode == 'raw' else 'raw'
                    mode_text = "EDGE DETECTION" if self.display_mode == 'edges' else "RAW FEEDS"
                    print(f"Display Mode: {mode_text}")
                elif key == ord('d'):
                    self.enable_detection = not self.enable_detection
                    status = "ENABLED" if self.enable_detection else "DISABLED"
                    print(f"Person Detection: {status}")
                elif key == ord('p'):
                    self.take_pictures = not self.take_pictures
                    status = "RECORDING" if self.take_pictures else "PAUSED"
                    print(f"Status: {status}")
                elif key == ord('t'):
                    self.current_colormap_idx = (self.current_colormap_idx + 1) % len(self.colormaps)
                    print(f"Thermal Colormap: {self.colormaps[self.current_colormap_idx][1]}")
                elif key in [ord(str(i)) for i in range(1, 10)]:
                    self.detection_interval = int(chr(key))
                    print(f"Detection Interval: Every {self.detection_interval} frame(s)")
                
                if not self.take_pictures:
                    cv2.waitKey(10)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            print("\nShutting down...")
            self.stop_event.set()
            
            rgb_process.join(timeout=2)
            thermal_process.join(timeout=2)
            
            if rgb_process.is_alive():
                rgb_process.terminate()
            if thermal_process.is_alive():
                thermal_process.terminate()
            
            cv2.destroyAllWindows()
            print("Complete!")

if __name__ == "__main__":
    Data_Collector().begin_data_stream()