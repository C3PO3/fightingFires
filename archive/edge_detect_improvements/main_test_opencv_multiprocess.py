"""
main_test_opencv_display_optimized.py - Multiprocessing optimized version
Uses separate processes for RGB and Thermal frame acquisition
Press 'q' to quit, 'e' to toggle edges, 'p' to pause
"""
import sys
import os
from mock_rgb import Rgb_Handler
from mock_thermal import Thermal_Handler
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
            
            # Non-blocking put - skip if queue full
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full
                
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
            
            # Non-blocking put - skip if queue full
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full
                
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
        self.rgb_queue = Queue(maxsize=2)  # Small queue to reduce latency
        self.thermal_queue = Queue(maxsize=2)
        self.stop_event = Event()
        
        # Current frames
        self.current_rgb = None
        self.current_thermal = None
        
        # Pre-compute edge detection kernel for reuse
        self.edge_kernel = (3, 3)
        
        # Thermal colormap
        self.colormaps = [
            (cv2.COLORMAP_JET, "JET"),
            (cv2.COLORMAP_HOT, "HOT"),
            (cv2.COLORMAP_INFERNO, "INFERNO"),
            (cv2.COLORMAP_TURBO, "TURBO")
        ]
        self.current_colormap_idx = 0
        
        # Create OpenCV windows
        cv2.namedWindow('RGB Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Thermal Feed', cv2.WINDOW_NORMAL)
        
        # Position windows side by side
        cv2.moveWindow('RGB Feed', 0, 0)
        cv2.moveWindow('Thermal Feed', 650, 0)
        
        # Resize windows for better display
        cv2.resizeWindow('RGB Feed', 640, 480)
        cv2.resizeWindow('Thermal Feed', 640, 480)
    
    def _get_latest_frames(self):
        """Get latest frames from queues (non-blocking)"""
        # Get RGB frame
        try:
            while not self.rgb_queue.empty():
                self.current_rgb = self.rgb_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Get Thermal frame
        try:
            while not self.thermal_queue.empty():
                self.current_thermal = self.thermal_queue.get_nowait()
        except queue.Empty:
            pass
    
    def _process_frame(self):
        """Process a single frame"""
        render_start = time.time()
        
        # Get latest frames from queues
        self._get_latest_frames()
        
        if self.current_rgb is None or self.current_thermal is None:
            return  # Wait for first frames
        
        rgb_frame = self.current_rgb
        thermal_frame = self.current_thermal
        
        # Save original (only if needed)
        rgb_frame_for_file_save = rgb_frame if self.save_images else None
        
        # Process RGB based on display mode
        if self.display_mode == 'edges':
            # Optimized edge detection pipeline
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, self.edge_kernel, 0)
            edges = cv2.Canny(blur, 50, 100)
            rgb_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            rgb_display = rgb_frame
        
        # Convert thermal to color (using current colormap)
        colormap, _ = self.colormaps[self.current_colormap_idx]
        thermal_display = cv2.applyColorMap(thermal_frame, colormap)
        
        # Add FPS overlay (only calculate when needed)
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Add text to both displays
            mode_text = "EDGES" if self.display_mode == 'edges' else "RAW"
            cv2.putText(rgb_display, f"FPS: {fps:.1f} | Mode: {mode_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(thermal_display, f"FPS: {fps:.1f} | Frames: {self.img_num}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frames
        cv2.imshow('RGB Feed', rgb_display)
        cv2.imshow('Thermal Feed', thermal_display)
        
        # Save images if enabled
        if self.save_images and self.take_pictures and rgb_frame_for_file_save is not None:
            filename_rgb = f"rgb{self.img_num}.jpg"
            filename_thm = f"thm{self.img_num}.jpg"
            save_path_rgb = os.path.join(self.usb_path_rgb, filename_rgb)
            save_path_thm = os.path.join(self.usb_path_thm, filename_thm)
            cv2.imwrite(save_path_rgb, rgb_frame_for_file_save)
            cv2.imwrite(save_path_thm, thermal_frame)
        
        if self.take_pictures:
            self.img_num += 1
        
        # Performance monitoring
        render_time = time.time() - render_start
        self.frame_times.append(render_time)
        
        # Print FPS every second
        if time.time() - self.last_fps_print >= 1.0:
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                mode_display = "EDGES" if self.display_mode == 'edges' else "RAW"
                colormap_name = self.colormaps[self.current_colormap_idx][1]
                print(f"FPS: {fps:.2f} | Frame Time: {avg_frame_time*1000:.2f}ms | "
                      f"Frames: {self.img_num} | Mode: {mode_display} | Colormap: {colormap_name}")
            self.frame_times = []
            self.last_fps_print = time.time()
    
    def begin_data_stream(self):
        print("\n" + "="*70)
        print("CAMERA SYSTEM - MULTIPROCESSING OPTIMIZED")
        print("="*70)
        print("Using separate processes for RGB and Thermal capture")
        print("")
        print("Keyboard Controls:")
        print("  'q' or ESC - Quit")
        print("  'e' - Toggle between RAW and EDGE detection")
        print("  'p' - Pause/Resume")
        print("  't' - Cycle thermal colormap")
        print("")
        print("Display mode: RAW FEEDS")
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
                
                # Handle keyboard input (1ms delay)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nQuitting...")
                    break
                elif key == ord('e'):
                    self.display_mode = 'edges' if self.display_mode == 'raw' else 'raw'
                    mode_text = "EDGE DETECTION" if self.display_mode == 'edges' else "RAW FEEDS"
                    print(f"Display Mode: {mode_text}")
                elif key == ord('p'):
                    self.take_pictures = not self.take_pictures
                    status = "RECORDING" if self.take_pictures else "PAUSED"
                    print(f"Status: {status}")
                elif key == ord('t'):
                    self.current_colormap_idx = (self.current_colormap_idx + 1) % len(self.colormaps)
                    print(f"Thermal Colormap: {self.colormaps[self.current_colormap_idx][1]}")
                
                # Always process a frame even when paused to keep windows responsive
                if not self.take_pictures:
                    cv2.waitKey(10)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            print("\nShutting down...")
            self.stop_event.set()
            
            # Wait for processes to finish
            rgb_process.join(timeout=2)
            thermal_process.join(timeout=2)
            
            # Force terminate if still alive
            if rgb_process.is_alive():
                rgb_process.terminate()
            if thermal_process.is_alive():
                thermal_process.terminate()
            
            cv2.destroyAllWindows()
            print("Complete!")

if __name__ == "__main__":
    Data_Collector().begin_data_stream()