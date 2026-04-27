# [README](https://github.com/C3PO3/fightingFires/blob/main/README.md)

## Overview
Instructions from Tufts Capstone Project Team (2025-2026) on how to set up software and hardware for the AI for FireFighters Project.

This comprehensive guide should take you step by step through:
   1. [Setting up a forward facing LiDAR](https://github.com/C3PO3/fightingFires/blob/main/Setting_up_LiDAR.md) to stream data frame by frame to be processed by other programs.
   2. [Training an AI model using Pointnet++](https://github.com/C3PO3/fightingFires/blob/main/Train_Pointnet%2B%2B.md) to do semantic segmentation for indoor environments. There are 13 classes in total, including beam, board, bookcase, ceiling, chair, clutter, column, door, floor, sofa, table, wall, and window.
   3. INSERT IMU INFO HERE

The goal of this guide is after you follow these setup instructions you should be able to work on integrating these components together, improve the accuracy of the AI model, and have a good jumping off point for your senior project.

## Hardware Requirements
To follow this setup guide make sure you have the following:
 - A windows laptop which can install wsl with ubuntu release 22.04. This is so it can support humble(for ROS2).
 - A forward facing LiDAR, supports ROS2.
   - Our LiDAR was: RoboSense E1R (forward-looking, solid-state “digital” LiDAR): Forward-looking FOV: 120° × 90° (very wide vertically, great for doors + people + obstacles), Range: up to ~75 m (more than enough indoors; even long corridors), Ruggedness: often listed as IP67 / IP6K9K and -40°C to +85°C (good for harsh environments like smoke/heat, within reason), ROS 2 ecosystem: RoboSense provides a driver/SDK that explicitly includes ROS2 support (rslidar_sdk)
 - VN-100 IMU - used for tracking location of the LiDAR needed for processing multiple frames at once.
 - TODO(AI team laptop requirmenets for training model)

## Folder Structure

