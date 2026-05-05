# Quickstart guide: Connecting and using the IMU

## Prerequisites:
- [VectorNav Control Center](https://www.vectornav.com/resources/software) is installed
- You have access to the VectorNav SDK:
	- [Request access](https://www.vectornav.com/resources/sdk) to the SDK. Click on "Request Access".
 	- I put "Tufts University" where it asks for company and "Engineer" for job title. Where it asks you to "describe your application", I put a brief summary of why we need the IMU for our project.
  	- I was granted access the same day I submitted the request form.
 
## Instructions
1. Connect the IMU into a power source and into your computer to transfer data. There are two connectors on the IMU board that you can connect the cables into. There is also a power brick that you can plug into the wall to power it. Once you've connected the IMU to power, the power light should come on (red).
2. Once you have the IMU powered on and connected to your computer, open the Control Center.
3. Go to the "Connect" tab. Select the COM port:
	- On Windows, open Device Manager.
	- Expand “Ports (COM & LPT)”.
	- Unplug and plug in the IMU to your computer.
	- Look for a new entry that appears. Select this COM port in the Control Center.
4. The default baud rate is 115200. You can change it if you want.
5. Click "Connect". Navigate to the "Views" tab. Now you can click on different views and see the metrics being displayed live. Views that may be of interest to you:
	- Yaw, Pitch, Roll
	- 3D View
	- Acceleration
	- Angular Rate
6. Your next step is to figure out how to use the VectorNav SDK to use the metrics in code. Then you can integrate the IMU and the LiDAR to get a pointcloud of a building.
	- Note: You will need to rigidly connect the two devices together. You may want to go to Nolop for this step.
