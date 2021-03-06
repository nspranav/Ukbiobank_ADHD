#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=176g
#SBATCH --gres=gpu:v100:4
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
python crossval_age_pred.py ${SLURM_JOBID}
