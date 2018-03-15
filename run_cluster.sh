#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o logs/out  # send stdout to sample_experiment_outfile
#SBATCH -e logs/err  # send stderr to sample_experiment_errfile
#SBATCH -t 8:00:00  # time requested in hour:minute:secon
#SBATCH --signal=USR1
#SBATCH --nodelist=landonia[12,17,21,23]

#To be used before srun so that interactive sessions are run with gpu support
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export LD_PRELOAD=/usr/lib64/libGLEW.so.1.10:$LD_PRELOAD

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mujoco

"$@"