"""
Performance testing script for camera system without hardware
Measures processing speed, frame rates, and bottlenecks
"""
import sys
import time
import cv2
import numpy as np
from collections import deque

# Import the mock handlers
from mock_rgb import Rgb_Handler
from mock_thermal import Thermal_Handler

class PerformanceMonitor:
    def __init__(self, window_size=60):
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.frame_count = 0
        
    def record_frame(self, processing_time):
        current_time = time.time()
        self.frame_times.append(current_time)
        self.processing_times.append(processing_time)
        self.frame_count += 1
    
    def get_stats(self):
        if len(self.frame_times) < 2:
            return None
        
        # Calculate FPS
        time_span = self.frame_times[-1] - self.frame_times[0]
        fps = len(self.frame_times) / time_span if time_span > 0 else 0
        
        # Calculate processing time stats
        avg_processing = np.mean(self.processing_times) * 1000  # ms
        max_processing = np.max(self.processing_times) * 1000
        min_processing = np.min(self.processing_times) * 1000
        
        # Calculate total runtime
        total_runtime = time.time() - self.start_time
        
        return {
            'fps': fps,
            'avg_processing_ms': avg_processing,
            'max_processing_ms': max_processing,
            'min_processing_ms': min_processing,
            'total_frames': self.frame_count,
            'total_runtime': total_runtime
        }

def test_camera_system(duration_seconds=10, display=True):
    """
    Test the camera system for a specified duration
    
    Args:
        duration_seconds: How long to run the test
        display: Whether to show the video feed
    """
    print("Initializing mock camera handlers...")
    rgb_handler = Rgb_Handler(2, 60)
    thermal_handler = Thermal_Handler(0, 60)
    
    monitor = PerformanceMonitor()
    
    print(f"\nRunning performance test for {duration_seconds} seconds...")
    print("Processing pipeline: Capture -> Grayscale -> Gaussian Blur -> Canny Edge Detection\n")
    
    start_time = time.time()
    last_print = start_time
    
    try:
        while (time.time() - start_time) < duration_seconds:
            frame_start = time.time()
            
            # Simulate the processing pipeline from main.py
            rgb_frame = rgb_handler.get_frame()
            thermal_frame = thermal_handler.get_frame()
            
            # RGB processing (as in main.py)
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(image=blur, threshold1=50, threshold2=100)
            
            # Thermal processing (convert to color for display)
            thermal_display = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)
            
            frame_end = time.time()
            processing_time = frame_end - frame_start
            
            monitor.record_frame(processing_time)
            
            # Print stats every second
            if time.time() - last_print >= 1.0:
                stats = monitor.get_stats()
                if stats:
                    print(f"Current FPS: {stats['fps']:.1f} | "
                          f"Avg Processing: {stats['avg_processing_ms']:.2f}ms | "
                          f"Frames: {stats['total_frames']}")
                last_print = time.time()
            
            # Display frames if requested
            if display:
                # Resize for display
                edges_resized = cv2.resize(edges, (400, 300))
                thermal_resized = cv2.resize(thermal_display, (400, 300))
                
                combined = np.hstack([
                    cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR),
                    thermal_resized
                ])
                
                # Add performance overlay
                stats = monitor.get_stats()
                if stats:
                    cv2.putText(combined, f"FPS: {stats['fps']:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, f"Process: {stats['avg_processing_ms']:.1f}ms", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, f"Frames: {stats['total_frames']}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Performance Test - RGB Edges | Thermal', combined)
                
                # Press 'q' to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nTest stopped by user")
                    break
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        # Clean up
        rgb_handler.release()
        thermal_handler.release()
        if display:
            cv2.destroyAllWindows()
        
        # Print final statistics
        stats = monitor.get_stats()
        if stats:
            print("\n" + "="*60)
            print("PERFORMANCE TEST RESULTS")
            print("="*60)
            print(f"Total Runtime:        {stats['total_runtime']:.2f} seconds")
            print(f"Total Frames:         {stats['total_frames']}")
            print(f"Average FPS:          {stats['fps']:.2f}")
            print(f"Avg Processing Time:  {stats['avg_processing_ms']:.2f} ms")
            print(f"Min Processing Time:  {stats['min_processing_ms']:.2f} ms")
            print(f"Max Processing Time:  {stats['max_processing_ms']:.2f} ms")
            print("="*60)
            
            # Performance assessment
            print("\nPerformance Assessment:")
            if stats['fps'] >= 25:
                print("✓ GOOD - Real-time capable (>25 FPS)")
            elif stats['fps'] >= 15:
                print("⚠ ACCEPTABLE - Minor lag possible (15-25 FPS)")
            else:
                print("✗ POOR - Significant lag expected (<15 FPS)")
            
            # Bottleneck analysis
            if stats['avg_processing_ms'] > 30:
                print("\n⚠ Processing time is high. Consider:")
                print("  - Reducing image resolution")
                print("  - Optimizing edge detection parameters")
                print("  - Using faster algorithms")

def benchmark_components(iterations=100):
    """
    Benchmark individual components of the processing pipeline
    """
    print("\n" + "="*60)
    print("COMPONENT BENCHMARK")
    print("="*60)
    print(f"Running {iterations} iterations per component\n")
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Benchmark grayscale conversion
    start = time.time()
    for _ in range(iterations):
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    gray_time = (time.time() - start) / iterations * 1000
    
    # Benchmark Gaussian blur
    gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    start = time.time()
    for _ in range(iterations):
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_time = (time.time() - start) / iterations * 1000
    
    # Benchmark Canny edge detection
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    start = time.time()
    for _ in range(iterations):
        edges = cv2.Canny(image=blur, threshold1=50, threshold2=100)
    canny_time = (time.time() - start) / iterations * 1000
    
    total_time = gray_time + blur_time + canny_time
    
    print(f"Grayscale Conversion:  {gray_time:.3f} ms")
    print(f"Gaussian Blur (3x3):   {blur_time:.3f} ms")
    print(f"Canny Edge Detection:  {canny_time:.3f} ms")
    print(f"-" * 40)
    print(f"Total Pipeline Time:   {total_time:.3f} ms")
    print(f"Theoretical Max FPS:   {1000/total_time:.1f}")
    print("="*60)

def run_stress_test(duration=30):
    """
    Run extended stress test to check for memory leaks or performance degradation
    """
    print("\n" + "="*60)
    print("STRESS TEST")
    print("="*60)
    print(f"Running extended test for {duration} seconds...")
    print("Monitoring for performance degradation and memory issues\n")
    
    test_camera_system(duration_seconds=duration, display=True)

if __name__ == "__main__":
    print("="*60)
    print("CAMERA SYSTEM PERFORMANCE TESTING TOOL")
    print("="*60)
    
    # Menu
    print("\nSelect test mode:")
    print("1. Quick Test (10 seconds)")
    print("2. Component Benchmark")
    print("3. Stress Test (30 seconds)")
    print("4. All Tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        test_camera_system(duration_seconds=10, display=True)
    elif choice == "2":
        benchmark_components()
    elif choice == "3":
        run_stress_test(duration=30)
    elif choice == "4":
        benchmark_components()
        test_camera_system(duration_seconds=10, display=True)
    else:
        print("Invalid choice. Running quick test...")
        test_camera_system(duration_seconds=10, display=True)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)