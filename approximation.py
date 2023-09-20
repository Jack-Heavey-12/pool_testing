# File that defines the approximation method for pool testing


import networkx as nx
import sys
import math
import numpy as np
import time
import pandas as pd
import json
import random


from collections import defaultdict
import itertools

from matplotlib import pyplot as plt

import time


def enumerate(graph, n_p=2):
	#returns list of the possible subsets that can be chosen
	#NOTE: This list is O(n^{n_p}), be aware of memory constraints
	nodes = list(graph.nodes())
	#set_list = []
	set_list = list(itertools.combinations(nodes, n_p))
	print('Potential Pools Enumerated')
	return set_list


def cascade_construction(graph, N, p, source_count=1):
	# Returns the list of connected components to the source.
	# NOTE: NOT a nx.Graph() type
	cascade_list = []

	#Generates one graph per sample
	for i in range(N):
		TempGraph = nx.Graph()

		TempGraph.add_nodes_from(graph.nodes())

		#Adds an edge if it is randomly selected with probability p
		for j in graph.edges():
			r = np.random.random()

			if r <= p:
				TempGraph.add_edge(j[0], j[1])

		#NOTE: May need to correct this later, but I think random.choose is the correct, will return a list
		#		Hopefully shouldn't be any assignment issues either

		src = random.choices(list(graph.nodes()), k=source_count).copy()
		ccs = []
		for source in src:
			ccs.append(list(nx.node_connected_component(TempGraph, source)))
		#Quick thing to remove duplicates. Think it should work fine
		ccs_f = []
		[ccs_f.append(x) for x in ccs if x not in ccs_f]
		cascade_list.append(ccs_f)
	print('Cascades Generated')
	return cascade_list

'''REQUIRED DATA TYPES FOR INPUTS:
	- pools: should be a list of tuples (sets?) of different pools, where nodes in pools are represented by integers
	- nodes: list of nodes (should all be of integer type just to make the enumeration easy)
	- cascades: list of cascades (connected component containing the source), order will remain constant
	- lam: integer(?) representing the lamda value in the LP (should this be in here or iterated)
	- epsilon: exogenous value that provides bounds
'''

def approximation(pools, nodes, cascades, lam=1.01, epsilon=.01, tau=1e-10):
	#first step is to construct the matrix A
	# 	- rows should be x(s) variables (pools) then stacked with v(i,s) 
	# 	- columns are the v(i,s)
	#	- 1 if node s is in the cascade, 0 otherwise

	#TODO
	# 1) Define delta, epsilon - DONE
	# 2) Figure out iterating for lambda, should that happen outside of the approximation function?
	# 3) Define stopping condition (think this is correct, while loop below)

	#First step is to generate the matrix A
	v_i_list = list(itertools.product(nodes, cascades))
	
	v_i_len = len(v_i_list)
	pool_len = len(pools)
	casc_len = len(cascades)

	A_xS = np.array([[1 if x in pools[0] else tauv for (x, _) in v_i_list]])

	#generates the x(S) rows related to the matrix A
	for i in pools[1:]:
		row = np.array([[1 if x in i else tau for (x, _) in v_i_list]])

		A_xS = np.vstack((A_xS, row))

	#generates the v(i,d) rows related to the matrix A

	A_vid = np.array([[1 if x == v_i_list[0] else tau for x in v_i_list]])
	for i in v_i_list[1:]:
		row = np.array([[1 if x == i else tau for x in v_i_list]])

		A_vid = np.vstack((A_vid, row))

	# A total vector, *HAS NOT* been transposed yet
	A = np.vstack((A_xS, A_vid))
	print(f'Shape of A: {A.shape}')
	#define the vectors c and b as defined in the dual program
	c_vec = np.array([1] * len(v_i_list))
	b_vec = np.array([1/casc_len] * v_i_len + [lam/casc_len] * pool_len)


	delta = (1 + epsilon) *  ((1+epsilon) * (v_i_len + pool_len)) ** (-1/epsilon)

	y_0 = delta / b_vec
	y = y_0
	primal = 0

	itera = 0

	while np.dot(b_vec, y) <= 1:
		# Do the iterations here
		length_vector = (A.T * np.atleast_2d(y)) # don't need to deal with the c vector because they're all 1, so dividing by c remains
		print(length_vector.shape)

		alpha_y = np.min(length_vector) ; q = np.argmin(length_vector) ;
		print(f'q: {q}, A shape: {A.shape}')
		min_capacity_edge_vec = b_vec / A[:, q]
		#print(min_capacity_edge_vec)

		min_capacity_edge = np.min(min_capacity_edge_vec) ; p = np.argmin(min_capacity_edge) ;

		#update of primal LP
		primal += b_vec[p] / A[p,q]

		#update of dual (our LP)
		y = y * (1 + epsilon * (b_vec[p] / A[p,q]) / (b_vec / np.squeeze(A[:, q])))
		itera += 1
		print(f'Current Iteration: {itera}')

	#break up the vector here, the final elements are the pools, the earlier elements are the node/cascade tuples
	x_s = np.array(y[-pool_len:])
	z_i_d = np.array(y[:-pool_len])

	# z = 1 - y, which means y = 1-z
	return x_s, 1-z_i_d





# current_full_y is a vector of the both the vector of cleared values and a sum with the pools
# Recall 1-y = z -> y = z+1
def lambda_search(current_full_y, prior_lam, len_pools, budget):
	x_s_vec = current_full_y[-len_pools:]

	new_lam = budget / sum(x_s_vec)

	# TODO


if __name__ == "__main__":
	start_time = time.time()
	#So this graph is only 75 nodes, 1138 edges
	print('Here we go!')
	df = pd.read_csv('data/hospital_contacts', sep='\t', header=None)
	df.columns = ['time', 'e1', 'e2', 'lab_1', 'lab_2']
	G = nx.from_pandas_edgelist(df, 'e1', 'e2')
	#G = nx.read_edgelist('data/test_graph.txt')

	mapping = dict(zip(G.nodes(),range(len(G))))
	graph = nx.relabel_nodes(G,mapping)
	
	set_list = enumerate(graph)
	cascade_list = cascade_construction(graph, 20, .05)

	#recall y is the full combined vector, so will need to split this up
	x_s, y_i_d = approximation(set_list, list(graph.nodes()), cascade_list, epsilon=.1)

	#x_prime = rounding(x)

	#rounded_obj_val = calculate_E_welfare(x_prime, cascade_list)

	print(f'Number of sets chosen: {np.sum(x_s)}')
	print(f'--- {time.time() - start_time} seconds ---')
	#print(f'LP Obj Val: {obj_value}, Rounded Obj Val: {rounded_obj_val}, size of x, y: {len(x)}, {len(y)}')




