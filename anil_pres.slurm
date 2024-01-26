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
	python for_anil.py 10
done

for i in {1..5}
do
	python for_anil.py 20
done


for i in {1..5}
do
	python for_anil.py 30
done


for i in {1..5}
do
	python for_anil.py 40
done


for i in {1..5}
do
	python for_anil.py 50
done
