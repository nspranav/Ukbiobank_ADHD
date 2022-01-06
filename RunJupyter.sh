#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=64g
#SBATCH --gres=gpu:v100:4
#SBATCH -p qTRDGPUL
#SBATCH -t 2800
#SBATCH -J Jupyter
#SBATCH -e error%A.err
#SBATCH -o out%A.out
#SBATCH -A PSYC0005
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nspranav1180@gmail.com
#SBATCH --oversubscribe

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo $HOSTNAME >&2 

source /home/users/pnadigapusuresh1/anaconda3/bin/activate CV2

jupyter notebook --no-browser --port=44444 --ip=0.0.0.0

