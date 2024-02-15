import networkx as nx
import sys
import math
import numpy as np
import time
import pandas as pd
import json
import random

import pickle


from collections import defaultdict
import itertools

def enumerate(graph, n_p=3):
	#returns list of the possible subsets that can be chosen
	#NOTE: This list is O(n^{n_p}), be aware of memory constraints
	nodes = list(graph.nodes())
	#set_list = []
	set_list = list(itertools.combinations(nodes, n_p))
	print('Potential Pools Enumerated')
	return set_list

def enumerate_random(graph, n_p, num_sets=1000):
	#Returns a shorter list of possible subsets for when graphs are larger
	#TODO
	nodes = list(graph.nodes())
	set_list = []
	for _ in range(num_sets):
		set_list.append(tuple(random.sample(nodes, n_p)))
	return set_list

# Copied over from original version of approximation.py with minor changes

def define_A_matrix(pools, nodes, cascades):
	# below product allows the cascade to be referenced by an integer, which we can use to index into the F_i dict
	v_i_list = list(itertools.product(nodes, list(range(len(cascades)))))


	#v_i_list_masked = list(itertools.product(list(range(len(nodes))), list(range(len(cascades)))))
	
	v_i_len = len(v_i_list)
	pool_len = len(pools)
	casc_len = len(cascades)


	#size of matrix is going to be pool_len + v_i_len rows, v_i_len for columns
	A = np.zeros((pool_len + v_i_len, v_i_len))

	approx_time = time.time()

	F_i = {} # Superset of F_vi, will define the sets here for each cascade. From here, only have to check for membership for the later constraint
	for i in range(len(cascades)):
		F_i[i] = [S for S in set_list if not any(x in cascades[i] for x in S)]


	for i in range(pool_len):
		#A[i+v_i_len, :] = np.array([0] * v_i_len)
		#A[i+v_i_len, :] = np.array([1 if (x not in casc[j]) for x in nodes for j in range(casc_len)])
		A[i+v_i_len, :] = np.array([1 if all((x in pools[i]), (pools[i] in F_i[j])) else 0 for (x, j) in v_i_list])

	A_vid = np.identity(v_i_len)

	for i in range(v_i_len):
		A[i] = A_vid[i, :]

	print(f'A Matrix Construction Time: {time.time() - approx_time} seconds ---')

	print(f'Shape of A: {A.shape}')

	c = np.array([0 if x in casc[j] else 1 for (x, j) in v_i_list])

	#doing this backwards, so want to transpose it eventually
	return A.T, c


def Approximation(graph, set_list, cascade_list, A, B=100, lam=2):
	''' inputs: 
		graph: nx.Graph()
		set_list: list of lists (output by either enumerate function)
		cascade_list: list of cascades (generally will be fixed and read in)
		A: numpy array, matrix defined from earlier function for the LP.
		B: integer for budget value
		lam: float, lambda guess
	'''

	v_i_list = list(itertools.product(nodes, cascades))
	
	v_i_len = len(v_i_list)
	pool_len = len(pools)
	casc_len = len(cascades)

	

	N = len(cascades)


## TODO - Actually update this, was just copied over, but a good skeleton for now
if __name__ == "__main__":

	graph = read_graph('uva_pre')
	
	if len(sys.argv) >= 2:
		num_sets = int(sys.argv[1])
	else:
		num_sets = 20000

	n_p = 4
	budget = 100

	set_list = enumerate_random(graph, n_p=n_p, num_sets=num_sets)

	fl = .33
	with open('test_cascades/uva_pre_1000_0.33.pkl', 'rb') as f:
		cascade_list = pickle.load(f)

	A, c = define_A_matrix(set_list, list(graph.nodes()), cascade_list)
	x, y, obj_value, variables = Approximation(graph, set_list, cascade_list, A, B=budget, lam=2)

	x_prime = rounding(x)
	nonzeros = {}
	for i in x.keys():
		if x[i] > 0:
			nonzeros[i] = x[i]
	print(f'Sets Chosen: {nonzeros}')


	rounded_obj_val = calculate_E_welfare(x_prime, cascade_list)

	#with open('results_file.csv', 'a') as f:
	#	f.write(f'\n{len(graph)},{len(cascade_list)},{fl},{obj_value},{rounded_obj_val}, Y, {len(set_list)}, {n_p},{budget}')
