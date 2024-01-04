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
	python pool.py 20
done
#============================================================================

for i in {1..5}
do
	python pool.py 22
done
#============================================================================

for i in {1..5}
do
	python pool.py 24
done
#============================================================================

for i in {1..5}
do
	python pool.py 26
done
#============================================================================

for i in {1..5}
do
	python pool.py 28
done
#============================================================================

for i in {1..5}
do
	python pool.py 30
done
#============================================================================
for i in {1..5}
do
	python pool.py 32
done
#============================================================================
for i in {1..5}
do
	python pool.py 34
done
#============================================================================
for i in {1..5}
do
	python pool.py 36
done
#============================================================================
for i in {1..5}
do
	python pool.py 38
done
#============================================================================
for i in {1..5}
do
	python pool.py 40
done
#============================================================================










