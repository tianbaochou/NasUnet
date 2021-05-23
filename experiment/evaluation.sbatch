#!/bin/bash
# Usage: sbatch slurm-gpu-job-script
# Prepared By: Ruibin Chen,  Apr 2021

# NOTE: To activate a SLURM option
# remove the whitespace between the '#' and 'SBATCH'

# Set your account id
#SBATCH --account=az20

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=3-00:00:00

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=RuibinJob

# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Request for GPU, 
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g

# SBATCH --gres=gpu:P100:1
# SBATCH --partition=m3h

# Memory usage (MB)
#SBATCH --mem-per-cpu=4000

# To receive an email when job completes or fails
#SBATCH --mail-user=rche0046@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=MY-DARTS-EVA-JOB-%j.out

# Set the file for error log (stderr)
#SBATCH --error=MY-DARTS-EVA-JOB-%j.err


# Use reserved node to run job when a node reservation is made for you already
# SBATCH --reservation=reservation_name


# Command to run a gpu job
# For example:

# module load cuda/7
module load anaconda/2019.03-Python3.7-gcc5

sleep 5
python -V

export PROJECT=az20
export CONDA_ENVS=/projects/$PROJECT/$USER/conda_envs
source activate $CONDA_ENVS/ruibinEnv
python -V

sleep 5
python ./MY-DARTS/train_ms.py --auxiliary --cutout --arch "Genotype(normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3)], normal_concat=[], reduce=[('sep_conv_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 3)], reduce_concat=[], up=[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3)], up_concat=[], dis=tensor([[2.2522, 2.2522, 2.2520, 2.2520],
        [2.2521, 2.2520, 2.2520, 2.2520],
        [2.2528, 2.2526, 2.2525, 2.2524],
        [2.2527, 2.2526, 2.2524, 2.2524],
        [2.2521, 2.2520, 2.2518, 2.2518]]), route=[[[[0, 0, 2], [1, 0, 2], [2, 0, 1], [3, 0, 2], [4, 1, 0], [5, 0, 3], [6, 0, 2], [7, 0, 1], [8, 0, 1]], [[0, 0, 2], [1, 0, 2], [2, 0, 1], [3, 0, 2], [4, 1, 0], [5, 0, 3], [6, 0, 2], [7, 0, 1], [8, 0, 2]], [[0, 0, 2], [1, 0, 2], [2, 0, 1], [3, 0, 2], [4, 1, 0], [5, 0, 3], [6, 0, 1], [7, 0, 1], [8, 0, 1]], [[0, 0, 2], [1, 0, 2], [2, 0, 1], [3, 0, 2], [4, 1, 0], [5, 0, 3], [6, 0, 2], [7, 0, 2], [8, 0, 1]]], [[[0, 0, 2], [1, 1, 0], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 2, 3], [8, 1, 3]], [[0, 0, 2], [1, 0, 2], [2, 0, 1], [3, 0, 2], [4, 1, 0], [5, 0, 3], [6, 0, 2], [7, 1, 0], [8, 1, 2]], [[0, 1, 0], [1, 1, 1], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 2, 3], [8, 1, 3]], [[0, 0, 2], [1, 1, 0], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 2, 1], [7, 2, 2], [8, 1, 3]]], [[[0, 0, 2], [1, 1, 0], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 2, 3]], [[0, 1, 0], [1, 1, 1], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 2, 3]], [[0, 0, 2], [1, 1, 0], [2, 1, 2], [3, 2, 0], [4, 2, 2], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 2, 3]], [[0, 1, 0], [1, 1, 2], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 2, 3]]], [[[0, 0, 2], [1, 1, 0], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 3, 2]], [[0, 1, 0], [1, 1, 1], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 3, 2]], [[0, 0, 2], [1, 1, 0], [2, 1, 2], [3, 2, 0], [4, 2, 2], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 3, 2]], [[0, 1, 0], [1, 1, 2], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 3, 2]]], [[[0, 0, 2], [1, 1, 0], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 4, 0]], [[0, 1, 0], [1, 1, 1], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 4, 0]], [[0, 0, 2], [1, 1, 0], [2, 1, 2], [3, 2, 0], [4, 2, 2], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 4, 0]], [[0, 1, 0], [1, 1, 2], [2, 1, 2], [3, 2, 0], [4, 2, 1], [5, 2, 2], [6, 3, 0], [7, 3, 1], [8, 4, 0]]]])"


nvidia-smi
# deviceQuery
