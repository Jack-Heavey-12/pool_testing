#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 36:00:00
#SBATCH	--mem-per-cpu=380000M
#SBATCH -p bii
#SBATCH -A nssac_students

module load anaconda
module load gurobi

export  PYTHONPATH=$EBROOTGUROBI/lib/python3.8_utf32

source activate lp

for i in {1..5}
do
	python pool_v1.py 5000
done
#============================================================================
for i in {1..5}
do
	python pool_v2.py 5000
done
