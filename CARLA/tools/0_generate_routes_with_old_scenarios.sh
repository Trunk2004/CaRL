#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-12:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/generate_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/generate_%a_%A.err
#SBATCH --partition=a100-galvani

# print info about current job
scontrol show job $SLURM_JOB_ID

echo "START TIME: $(date)"

export SCENARIO_RUNNER_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

python -u generate_long_routes_with_scenarios.py --save_folder /mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/leaderboard/data/1000_meters_old_scenarios_01 --carla_root /mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15 --start_repetition $SLURM_ARRAY_TASK_ID --scenario_dilation 100 --scenario_runner_root /mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/scenario_runner --generate_scenarios 1 --only_leaderboard_1 1 --route_length 1000


echo "END TIME: $(date)"
