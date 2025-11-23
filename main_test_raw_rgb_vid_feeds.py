"""
main_test_raw_feeds.py - Display raw RGB and thermal camera feeds
Shows actual camera output instead of edge detection
Toggle between raw feeds and edge detection with a button
"""
import sys
import os
from mock_rgb import Rgb_Handler
from mock_thermal import Thermal_Handler
import cv2
import time
from tkinter import *
from PIL import Image, ImageTk

class Data_Collector():
    def __init__(self):
        # Boolean for taking or stop taking live pictures
        self.take_pictures = True
        
        # Initialize handlers for RGB and thermal
        self.rgb_handler = Rgb_Handler(2, 60)
        self.thermal_handler = Thermal_Handler(0, 60)
        
        # GUI member variables
        self.window = Tk()
        self.window.title("Camera GUI - RAW FEEDS MODE")
        
        # Create a label for displaying the video feed
        self.label_rgb = Label(self.window, width=300, height=300)
        self.label_rgb.pack(side=LEFT, padx=10, pady=10)
        self.label_thermal = Label(self.window, width=300, height=300)
        self.label_thermal.pack(side=LEFT, padx=10, pady=10)
        
        # Buttons
        self.record_button = None
        self.mode_button = None
        
        # Image saving information (disabled for testing)
        self.img_num = 0
        self.save_images = False
        self.usb_path_rgb = "./test_output/rgb_images"
        self.usb_path_thm = "./test_output/thm_images"
        
        # Create output directories if they don't exist
        if self.save_images:
            os.makedirs(self.usb_path_rgb, exist_ok=True)
            os.makedirs(self.usb_path_thm, exist_ok=True)
        
        # Performance monitoring
        self.frame_times = []
        self.last_fps_print = time.time()
        
        # Display mode: 'raw' or 'edges'
        self.display_mode = 'raw'  # Start with raw feeds
    
    def _toggle_record(self):
        self.take_pictures = not self.take_pictures
        status = "RECORDING" if self.take_pictures else "PAUSED"
        print(f"Status: {status}")
    
    def _toggle_display_mode(self):
        """Toggle between raw feeds and edge detection"""
        self.display_mode = 'edges' if self.display_mode == 'raw' else 'raw'
        mode_text = "EDGE DETECTION" if self.display_mode == 'edges' else "RAW FEEDS"
        print(f"Display Mode: {mode_text}")
        self.window.title(f"Camera GUI - {mode_text}")
    
    def _render(self):
        if self.take_pictures:
            render_start = time.time()
            
            # Take the RGB and thermal frame
            rgb_frame = self.rgb_handler.get_frame()
            thermal_frame = self.thermal_handler.get_frame()
            
            # Save original for file save
            rgb_frame_for_file_save = rgb_frame.copy()
            
            # Process RGB based on display mode
            if self.display_mode == 'edges':
                # Apply edge detection
                gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                edges = cv2.Canny(image=blur, threshold1=50, threshold2=100)
                rgb_frame_display = edges
                is_grayscale = True
            else:
                # Show raw RGB feed
                rgb_frame_display = rgb_frame
                is_grayscale = False
            
            # Convert thermal to color for display
            thermal_frame_color = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)
            thermal_frame_color = thermal_frame_color[:,:,::-1]  # BGR to RGB
            
            # Prepare RGB image for display
            if is_grayscale:
                # Edge detection output is already grayscale
                rgbimg = Image.fromarray(rgb_frame_display)
            else:
                # Convert BGR to RGB for display
                rgb_frame_rgb = cv2.cvtColor(rgb_frame_display, cv2.COLOR_BGR2RGB)
                rgbimg = Image.fromarray(rgb_frame_rgb)
            
            # Resize images
            rgbimg = rgbimg.resize((400, 300))
            rgbimgtk = ImageTk.PhotoImage(image=rgbimg)
            
            thermalimg = Image.fromarray(thermal_frame_color)
            thermalimg = thermalimg.resize((400, 300))
            thermalimgtk = ImageTk.PhotoImage(image=thermalimg)
            
            # Update the labels with the new images
            self.label_rgb.imgtk = rgbimgtk
            self.label_rgb.config(image=rgbimgtk)
            self.label_thermal.imgtk = thermalimgtk
            self.label_thermal.config(image=thermalimgtk)
            
            # Save the images (if enabled)
            if self.save_images:
                filename_rgb = f"rgb{self.img_num}.jpg"
                filename_thm = f"thm{self.img_num}.jpg"             
                save_path_rgb = os.path.join(self.usb_path_rgb, filename_rgb)
                save_path_thm = os.path.join(self.usb_path_thm, filename_thm)
                cv2.imwrite(save_path_rgb, rgb_frame_for_file_save)
                cv2.imwrite(save_path_thm, thermal_frame)
            
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
                    print(f"FPS: {fps:.2f} | Frame Time: {avg_frame_time*1000:.2f}ms | Frames: {self.img_num} | Mode: {mode_display}")
                self.frame_times = []
                self.last_fps_print = time.time()
        
        # Schedule next render
        self.window.after(10, self._render)
    
    def begin_data_stream(self):
        # Create control buttons
        button_frame = Frame(self.window)
        button_frame.pack(pady=10)
        
        self.record_button = Button(
            button_frame, 
            text='START/STOP', 
            height=2, 
            width=15, 
            command=self._toggle_record,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.record_button.pack(side=LEFT, padx=5)
        
        self.mode_button = Button(
            button_frame,
            text='TOGGLE EDGES',
            height=2,
            width=15,
            command=self._toggle_display_mode,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.mode_button.pack(side=LEFT, padx=5)
        
        # Print startup info
        print("\n" + "="*70)
        print("CAMERA SYSTEM - RAW FEEDS MODE")
        print("="*70)
        print("Using fast mock RGB and thermal camera handlers")
        print("Image saving:", "ENABLED" if self.save_images else "DISABLED")
        print("")
        print("Controls:")
        print("  - START/STOP: Pause/resume video feed")
        print("  - TOGGLE EDGES: Switch between raw feeds and edge detection")
        print("  - Close window to exit")
        print("")
        print("Display mode: RAW FEEDS (showing actual camera output)")
        print("="*70 + "\n")
        
        # Start rendering
        self._render()
        self.window.mainloop()
        
        # Cleanup
        print("\nShutting down...")
        self.rgb_handler.release()
        self.thermal_handler.release()
        print("Complete!")

if __name__ == "__main__":
    Data_Collector().begin_data_stream()