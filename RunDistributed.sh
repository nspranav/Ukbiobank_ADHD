#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=70g
#SBATCH --gres=gpu:V100:2
#SBATCH -p qTRDGPUH
#SBATCH --nodelist=arctrdgn002
#SBATCH -J ConvLr00001
#SBATCH -e error%A.err
#SBATCH -o gpu4_%A.txt
#SBATCH -A trends396s109
#SBATCH -t 0-05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe


source /data/users2/pnadigapusuresh1/software/bin/activate latest
python compute_attributions.py ${SLURM_JOBID}
