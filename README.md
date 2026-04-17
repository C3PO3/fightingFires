# [README](https://github.com/C3PO3/fightingFires/blob/main/README.md)

## Overview

## Hardware Requirements
To follow this setup guide make sure you have the following:
 - A windows laptop which can install wsl with ubuntu release 22.04. This is so it can support humble(for ROS2).
 - A forward facing LiDAR, supports ROS2.
   - Our LiDAR was: RoboSense E1R (forward-looking, solid-state “digital” LiDAR): Forward-looking FOV: 120° × 90° (very wide vertically, great for doors + people + obstacles), Range: up to ~75 m (more than enough indoors; even long corridors), Ruggedness: often listed as IP67 / IP6K9K and -40°C to +85°C (good for harsh environments like smoke/heat, within reason), ROS 2 ecosystem: RoboSense provides a driver/SDK that explicitly includes ROS2 support (rslidar_sdk)
 - VN-100 IMU - used for tracking location of the LiDAR needed for processing multiple frames at once.
 - TODO(AI team laptop requirmenets for training model)

## Running tests


## Folder Structure

