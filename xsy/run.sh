#!/bin/bash

# start fk pose
python fk_pose.py \
    --device cpu

# # start play robot control animation
# python scripts/rsl_rl/play.py \
#     --task Unitree-G1-29dof-Velocity
