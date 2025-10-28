#!/bin/bash
export SCENARIO_RUNNER_ROOT=/home/trung/CaRL/CARLA/original_leaderboard/scenario_runner 
export LEADERBOARD_ROOT=/home/trung/CaRL/CARLA/original_leaderboard/leaderboard
export CARLA_ROOT=/home/trung/CaRL/CARLA/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export WORK_DIR=/home/trung/CaRL/CARLA

export DEBUG_ENV_AGENT=1 # Produce debug outputs
export SAVE_PATH=/home/trung/Videos # Folder to save debug output in
export RECORD=0 # Record infraction clips
export SAVE_PNG=1 # Save higher quality individual debug frames in PNG. Otherwise video is saved. 
export UPSCALE_FACTOR=1 
export CUBLAS_WORKSPACE_CONFIG=:4096:8 

export PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch 

python ${WORK_DIR}/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${WORK_DIR}/custom_leaderboard/leaderboard/data/longest6.xml --agent ${WORK_DIR}/team_code/eval_agent.py --resume 1 --track MAP --port 2000 --traffic-manager-port 8000 --agent-config /home/trung/CaRL/CARLA/results/CaRL_PY_01 --record True
