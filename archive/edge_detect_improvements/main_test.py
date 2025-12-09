#copy of main that uses our mock python files for now

import sys
import os
# from seg import h_seg  # Commented out - not needed for testing
from mock_rgb import Rgb_Handler  # Changed to mock
from mock_thermal import Thermal_Handler  # Changed to mock
import cv2
import select
import time
from tkinter import *
from PIL import Image, ImageTk

class Data_Collector():
    def __init__(self):
        # Boolean for taking or stop taking live pictures
        self.take_pictures = True
        
        # Initialize handlers for RGB and thermal to their respective indices
        self.rgb_handler = Rgb_Handler(2, 60)
        self.thermal_handler = Thermal_Handler(0, 60)  # Fixed: was Rgb_Handler in original
        
        # GUI member variables
        self.window = Tk()
        self.window.title("Camera GUI - TESTING MODE")
        
        # Create a label for displaying the video feed
        self.label_rgb = Label(self.window, width=300, height=300)
        self.label_rgb.pack(side=LEFT, padx=10, pady=10)
        self.label_thermal = Label(self.window, width=300, height=300)
        self.label_thermal.pack(side=LEFT, padx=10, pady=10)
        
        # Make button now. Initialize later
        self.record_button = None 
        
        # Image saving information (disabled for testing)
        self.img_num = 0
        self.save_images = False  # Added flag to control image saving
        self.usb_path_rgb = "./test_output/rgb_images"  # Changed to local path
        self.usb_path_thm = "./test_output/thm_images"  # Changed to local path
        
        # Create output directories if they don't exist
        if self.save_images:
            os.makedirs(self.usb_path_rgb, exist_ok=True)
            os.makedirs(self.usb_path_thm, exist_ok=True)
        
        # Performance monitoring
        self.frame_times = []
        self.last_fps_print = time.time()
    
    def _button_callback(self, channel):
        self.take_pictures = not (self.take_pictures)
    
    def _toggle_record(self):
        self.take_pictures = not self.take_pictures
        status = "RECORDING" if self.take_pictures else "PAUSED"
        print(f"Status: {status}")
    
    def _render(self):
        if self.take_pictures:
            render_start = time.time()
            
            # Take the RGB and thermal frame
            rgb_frame = self.rgb_handler.get_frame()
            thermal_frame = self.thermal_handler.get_frame()
            
            # Convert thermal to color for display (it's grayscale from handler)
            thermal_frame_color = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)
            thermal_frame_color = thermal_frame_color[:,:,::-1]  # BGR to RGB
            
            rgb_frame_for_file_save = rgb_frame.copy()
            
            # Perform canny edge detection
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(image=blur, threshold1=50, threshold2=100)
            rgb_frame_display = edges
            
            # Resize and reformat the thermal and rgb images for GUI
            rgbimg = Image.fromarray(rgb_frame_display)
            rgbimg = rgbimg.resize((400, 300))
            rgbimgtk = ImageTk.PhotoImage(image=rgbimg)
            
            thermalimg = Image.fromarray(thermal_frame_color)
            thermalimg = thermalimg.resize((400, 300))
            thermalimgtk = ImageTk.PhotoImage(image=thermalimg)
            
            # Update the label with the new image
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
                    print(f"FPS: {fps:.2f} | Avg Frame Time: {avg_frame_time*1000:.2f}ms | Frames: {self.img_num}")
                self.frame_times = []
                self.last_fps_print = time.time()
        
        self.window.after(10, self._render)
    
    def begin_data_stream(self):
        self.record_button = Button(
            self.window, 
            text='START/STOP', 
            height=5, 
            width=10, 
            command=self._toggle_record
        )
        self.record_button.pack()
        
        print("\n" + "="*60)
        print("CAMERA SYSTEM - TESTING MODE")
        print("="*60)
        print("Using mock RGB and thermal camera handlers")
        print("Image saving:", "ENABLED" if self.save_images else "DISABLED")
        print("Press START/STOP button to pause/resume")
        print("Close window to exit")
        print("="*60 + "\n")
        
        self._render()
        self.window.mainloop()
        
        # Cleanup
        print("\nShutting down...")
        self.rgb_handler.release()
        self.thermal_handler.release()
        print("Complete!")

if __name__ == "__main__":
    Data_Collector().begin_data_stream()