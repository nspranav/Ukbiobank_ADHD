#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=500g
#SBATCH --gres=gpu:V100:1
#SBATCH -t 3-00:00
#SBATCH -p qTRDHM
#SBATCH -J Jupyter
#SBATCH -e error%A.err
#SBATCH -o gpu4_%A.txt
#SBATCH -A trends396s109
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe

echo $HOSTNAME >&2 

source /data/users2/pnadigapusuresh1/software/bin/activate latest

jupyter lab --no-browser --port=44444 --ip=0.0.0.0

sleep 30s

