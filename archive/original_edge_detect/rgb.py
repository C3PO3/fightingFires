import cv2
import time
import sys

class Rgb_Handler:

    def __init__(self, port_index = 1, fps_cap = 60):
        self.cap = cv2.VideoCapture(port_index)
        self.cap.set(5, fps_cap)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        self.framecount = 0
        self.prevMillis = 0

        print(self.cap.get(5))

    def print_fps(self):
        millis = int(round(time.time() * 1000))
        self.framecount += 1
        if millis - self.prevMillis > 1000:
            print(self.framecount)
            self.prevMillis = millis 
            self.framecount = 0
    
    def get_frame(self):
        ret, frame = self.cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #blur = cv2.blur(frame, (5, 5))
        #ret, thresh = cv2.threshold(blur, 170, 255, 0)
        if not ret:
            print("Can't receive frame (stream end?).")
            sys.exit(1)

        # cv2.imshow("frame", frame)
        return frame
    
    def release (self):
        self.cap.release()