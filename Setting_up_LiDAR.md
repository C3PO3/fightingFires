# How to connect RoboSense LiDAR to your Windows laptop

## Prerequisites:

 - Have a windows laptop with WSL installed. If not installed, open your terminal as an administrator and run `wsl --install` to install it
 - Have a windows laptop and install wsl using ubuntu release 22.04 (specifically this version as newer versions do not support ROS2 Humble)
    - To set this up run: `wsl --install -d Ubuntu-22.04`, then `wsl --set-default Ubuntu-22.04`
    - Now when you run wsl you should be using the correct release
    - To verify this inside wsl run `lsb_release -a` and you should see ubuntu's release listed as 22.04
 - Optional: Have wireshark installed to view packet flow

## Instructions:

1. If your Windows laptop does not have an ethernet port, you will first need to plug in an ethernet adapter. This is so your settings will display properly.
2. Go to Settings > Network & internet > Ethernet. Change the IP settings from automatic to manual. Input the following settings:
IPv4 (toggled on)
IP address: 192.168.1.102
Subnet mask: 255.255.255.0
Gateway: 192.168.1.1
Preferred DNS: 8.8.8.8
DNS over HTTPS: Off

Note that you may have trouble changing the IP address if the router on your home network uses the same subnet mask. I ran into this issue, so I went to on-campus buildings and used Tufts_Secure when working with the LiDAR.

3. Plug in the lidar into the ethernet port (or adapter)
4. As a sanity check open wireshark and click ethernet (or ethernet 2) you should see packets flowing in from the LiDAR to destination 192.168.1.102.
5. [Install rslidar_sdk](https://github.com/RoboSense-LiDAR/rslidar_sdk) follow the instructions in the readme. Make sure you install ROS2 humble when it gives you the option of which version of ROS2 to install.
 - NOTE: Before building, edit src/rslidar_sdk/config/config.yaml file to make sure LiDAR model is right: RSE1 Check that msop_port and difop_port are right: 
   msop_port: 6699              #  Msop port of lidar
   difop_port: 7788             #  Difop port of lidar
6. Once done with that turn off Windows Defender Firewall (needed so ubuntu can see the packets)(turn back on when not working with the LiDAR)
7. In your windows terminal(outside wsl) run: `notepad %USERPROFILE%\.wslconfig` and edit the config file to say:
[wsl2]
networkingMode=mirrored
8. Now restart wsl with the updated config file: `wsl --shutdown`
9. Run rviz: `ros2 launch rslidar_sdk start.py`
10. Check traffic is flowing in by running: `ss -uln` and check for port 6699
11. Now run: `ros2 run tf2_tools view_frames` and in your rviz you should see the frames displayed in 3D

