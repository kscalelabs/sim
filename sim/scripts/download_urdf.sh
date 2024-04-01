#!/bin/zsh
# ./sim/scripts/download_urdf.sh

# URL of the latest model.
url=https://cad.onshape.com/documents/71f793a23ab7562fb9dec82d/w/6160a4f44eb6113d3fa116cd/e/1a95e260677a2d2d5a3b1eb3

# Output directory.
output_dir=${MODEL_DIR}/robots/stompy

kol urdf ${url} \
    --max-ang-velocity 31.4 \
    --suffix-to-joint-effort \
        dof_x4_h=1.5 \
        dof_x4=1.5 \
        dof_x6=3 \
        dof_x8=6 \
        dof_x10=12 \
        knee_revolute=13.9 \
        ankle_revolute=6 \
    --output-dir ${output_dir}
