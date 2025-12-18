# README

## Running tests
These tests expect you to be in the root folder so inside fightingFires
 - to run edge detect: python current_optimization_tests\current_main.py
 - to run edge detect with person detection: python person_recognition\main_with_person_detection.py

## Folder Structure
optimized_edge_detect:
 - contains our usable version of edge detect
 - Performance 200+ FPS
person_recognition:
 - currently under test(INCOMPLETE)
 - using optimized_edge_detect to recognize people and output people and rough coordinates of the people's location
archive:
 - original_edge_detect:
    - Original code forked from: https://github.com/axu04/fightingFires
 - edge_detect_improvements:
    - Saved work partially improved version of original code. Not as good as the current version in optimized_edge_detect but saved for documenting progress


