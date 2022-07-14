#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=44g
#SBATCH --gres=gpu:v100:1
#SBATCH -p qTRDGPUH
#SBATCH -t 7680
#SBATCH -J Jupyter
#SBATCH -e error%A.err
#SBATCH -o out%A.out
#SBATCH -A PSYC0005
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe

echo $HOSTNAME >&2 

source /home/users/pnadigapusuresh1/anaconda3/bin/activate latest

jupyter notebook --no-browser --port=44444 --ip=0.0.0.0

