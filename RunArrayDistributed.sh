#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=88g
#SBATCH --gres=gpu:v100:2
#SBATCH -p qTRDGPUH
#SBATCH -t 7680
#SBATCH -J ConvLr00001
#SBATCH -e error%A.err
#SBATCH -o gpu4_%A.txt
#SBATCH -A PSYC0005
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe


source /home/users/pnadigapusuresh1/anaconda3/bin/activate latest
python distributed_conv.py ${SLURM_ARRAY_TASK_ID}


export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo $HOSTNAME >&2

module load Framework/Matlab2019b
matlab -batch 'array_example($SLURM_ARRAY_TASK_ID)'

sleep 30s