import sys
import os
from seg import h_seg
from rgb import Rgb_Handler
from thermal import Thermal_Handler
import cv2
import select
import time
from tkinter import *
from PIL import Image, ImageTk

class Data_Collector():
    def __init__(self):
        # Boolean for taking or stop taking live pictures
        self.take_pictures = True
        
        # Intialize handlers for RGB and thermal to their respective indices
        self.rgb_handler = Rgb_Handler(2, 60)
        self.thermal_handler = Rgb_Handler(0, 60)

        # GUI member variables
        self.window = Tk()
        self.window.title("Camera GUI")

        # Create a label for displaying the video feed
        self.label_rgb = Label(self.window, width = 300, height= 300)
        self.label_rgb.pack(side = LEFT, padx=10, pady=10)
        self.label_thermal = Label(self.window, width = 300, height= 300)
        self.label_thermal.pack(side = LEFT,padx=10, pady=10)

        # Make button now. Initialize later
        self.record_button = None 
        
        # Image saving information
        self.img_num = 0
        self.usb_path_rgb = "/media/pi/DCA5-2D88/rgb_images"
        self.usb_path_thm = "/media/pi/DCA5-2D88/thm_images"

    def _button_callback(self, channel):
        self.take_pictures = not (self.take_pictures)
    
    def _toggle_record(self):
        self.take_pictures = not self.take_pictures
    
    def _render (self):
        if (self.take_pictures == True):
            # Take the RGB and thermal frame
            rgb_frame = self.rgb_handler.get_frame()
            thermal_frame = self.thermal_handler.get_frame()
            thermal_frame = thermal_frame[:,:,::-1]
            rgb_frame_for_file_save = rgb_frame
            
            # Perform canny edge detection
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            edges = cv2.Canny(image=blur, threshold1=50, threshold2=100)
            rgb_frame = edges
            
            # Resize and reformat the thermal and rgb images for GUI
            rgbimg = Image.fromarray(rgb_frame)
            rgbimg = rgbimg.resize((400, 300))
            rgbimgtk = ImageTk.PhotoImage(image=rgbimg)
            thermalimg = Image.fromarray(thermal_frame)
            thermalimg = thermalimg.resize((400, 300))
            thermalimgtk = ImageTk.PhotoImage(image=thermalimg)

            # Update the label with the new image
            self.label_rgb.imgtk = rgbimgtk
            self.label_rgb.config(image=rgbimgtk)
            self.label_thermal.imgtk = thermalimgtk
            self.label_thermal.config(image=thermalimgtk)
            
            # Save the images
            filename_rgb = f"rgb{self.img_num}.jpg"
            filename_thm = f"thm{self.img_num}.jpg"             
            save_path_rgb = os.path.join(self.usb_path_rgb, filename_rgb)
            save_path_thm = os.path.join(self.usb_path_thm, filename_thm)
            # TODO For future developers: Uncomment below two lines to save images
            #cv2.imwrite(save_path_rgb, rgb_frame_for_file_save)
            #cv2.imwrite(save_path_thm, thermal_frame)
            self.img_num+=1
        self.window.after(10, self._render)

    def begin_data_stream(self):
        self.record_button = Button(self.window, text = 'START/STOP', height=5, width=10, command = self._toggle_record)
        self.record_button.pack()

        self._render()

        self.window.mainloop()

Data_Collector().begin_data_stream()