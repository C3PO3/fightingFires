# How to connect and use a RoboSense LiDAR with a Windows laptop

## Prerequisites:

 - Have a windows laptop and install wsl using ubuntu release 22.04 (specifically this version as newer versions do not support ROS2 Humble)
    - To set this up run: `wsl --install -d Ubuntu-22.04`, then `wsl --set-default Ubuntu-22.04`
    - Now when you run wsl you should be using the correct release
    - To verify this inside wsl run `lsb_release -a` and you should see ubuntu's release listed as 22.04
 - If your windows laptop does not have an ethernet port, you will need an ethernet adapter.
 - Optional: Have Wireshark installed to observe packet flow.

## Instructions for setting up the LiDAR:

1. If your Windows laptop does not have an ethernet port, you will first need to plug in an ethernet adapter. This is so your settings will display properly.
2. Go to Settings > Network & internet > Ethernet. Change the IP settings from automatic to manual. Input the following settings:
IPv4 (toggled on)
IP address: 192.168.1.102
Subnet mask: 255.255.255.0
Gateway: 192.168.1.1
Preferred DNS: 8.8.8.8
DNS over HTTPS: Off

Note that you may have trouble changing the IP address if your home network is using the same IP range (192.168.1.xxx). I ran into this issue, so I went to on-campus buildings and used Tufts_Secure when working with the LiDAR.

3. Make sure the LiDAR has power by plugging in the power cord into the LiDAR and into the wall. Plug in the yellow ethernet cord to the LiDAR and into your computer (or adapter).
4. As a sanity check open Wireshark and click on ethernet (or ethernet 2). You should see packets flowing in from the LiDAR (192.168.1.200) to destination 192.168.1.102 using port 6699.
5. [Install rslidar_sdk](https://github.com/RoboSense-LiDAR/rslidar_sdk) by following the instructions in the readme. Make sure you [install ROS2 humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) in step 3 when it asks you to install ROS or ROS2.
 - NOTE: Before building, edit src/rslidar_sdk/config/config.yaml so these variables are correct:
   ```
   lidar_type: RSE1             #  LiDAR type
   msop_port: 6699              #  Msop port of lidar
   difop_port: 7788             #  Difop port of lidar
   ```
6. Turn off Windows Defender Firewall. I only had to turn off the firewall under "Public network settings", but this may change based on what network you on. Turning off the firewall is needed so that Ubuntu can see the UDP packets being received from the LiDAR. Make sure to turn back on the firewall when you're done using the LiDAR.
7. Go to File Explorer and find the wsl config file. The path will be something like C:\Users\<your_username>\.wslconfig. Edit it to say:
```
[wsl2]
networkingMode=mirrored
```
8. Restart wsl with the updated config file: run `wsl --shutdown` and then `wsl`.
9. Source ROS2 humble and the SDK build (replacing lidar_ws with the name of your workspace folder):
```
source /opt/ros/humble/setup.bash
source ~/lidar_ws/install/setup.bash
```
- NOTE: You will need to run these commands in every new Ubuntu terminal before you can run ros2. You may want to add these to a startup script that runs whenever you open a new Ubuntu terminal, but I haven't done this myself.
10. Run rviz: `ros2 launch rslidar_sdk start.py`. You should see a window pop up displaying a 3D coordinate space and an XY plane.
11. We can check that port 6699 is listening: `ss -uln` and check for port 6699.
12. Now run: `ros2 run tf2_tools view_frames` and in your rviz window you should see the frames being displayed real-time in the 3D coordinate space.

## Instructions for capturing frames in c++
1. Create a new directory inside src called frame_inspector. This will be the name of the package we are creating.
2. Copy the files and file structure from the frame_inspector folder of this repo into your frame_inspector directory. Now the file structure of your lidar workspace should look like this:
```
lidar_ws
├── build
├── install
├── log
└── src
    ├── rslidar_sdk
    ├── rslidar_msg
    └── frame_inspector
        ├── CMakeLists.txt
        ├── package.xml
        └── src
            └── pointcloud_printer.cpp
```
3. cd into ~/lidar_ws and run `colcon build`.
4. Run `ros2 run frame_inspector pointcloud_printer`. This will run the pointcloud_printer executable from the frame_inspector package we just made. Note: you may want to redirect the output to a file.
5. Now you have a program that prints out each frame's coordinates. The next step would be to integrate this program with the IMU to mesh many frames together over time to form a pointcloud of an entire building.
