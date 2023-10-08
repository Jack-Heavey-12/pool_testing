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

def define_A_matrix(pools, nodes, cascades, tau=1e-10):
	v_i_list = list(itertools.product(nodes, cascades))
	
	v_i_len = len(v_i_list)
	pool_len = len(pools)
	casc_len = len(cascades)


	#size of matrix is going to be pool_len + v_i_len rows, v_i_len for columns
	A = np.zeros((pool_len + v_i_len, v_i_len))

	approx_time = time.time()

	for i in range(pool_len):
		A[i, :] = np.array([1 if ((x in pools[i]) and (x not in casc)) else tau for (x, casc) in v_i_list])
		#A[i, :] = np.array([1 if (x in pools[i]) else tau for (x, casc) in v_i_list])

	
	#A_xS = np.array([[1 if x in pools[0] else tau for (x, _) in v_i_list]])

	#generates the x(S) rows related to the matrix A
	#for i in pools[1:]:
	#	row = np.array([[1 if x in i else tau for (x, _) in v_i_list]])

	#	A_xS = np.vstack((A_xS, row))

	#generates the v(i,d) rows related to the matrix A

	'''A_vid = np.array([[1 if x == v_i_list[0] else tau for x in v_i_list]])
	for i in v_i_list[1:]:
		row = np.array([[1 if x == i else tau for x in v_i_list]])

		A_vid = np.vstack((A_vid, row))'''
	A_vid = np.identity(v_i_len)
	inds = A_vid == 0
	A_vid[inds] = tau

	for i in range(v_i_len):
		A[i + pool_len] = A_vid[i, :]
	
	# A total vector, *HAS NOT* been transposed yet
	#A_xS = np.vstack((A_xS, A_vid))
	print(f'A Matrix Construction Time: {time.time() - approx_time} seconds ---')

	print(f'Shape of A: {A.shape}')

	return A




def approximation(A, pools, nodes, cascades, lam=1.01, epsilon=.01, tau=1e-10):
	#first step is to construct the matrix A
	# 	- rows should be x(s) variables (pools) then stacked with v(i,s) 
	# 	- columns are the v(i,s)
	#	- 1 if node s is in the cascade, 0 otherwise

	#TODO
	# 1) Define delta, epsilon - DONE
	# 2) Figure out iterating for lambda, should that happen outside of the approximation function?
	# 3) Define stopping condition (think this is correct, while loop below)
	
	# setup the binary search subroutine here
	lam_array = list(range(1, len(nodes) ** 2 + 1))
	mid_index = len(lam_array) // 2
	initial_lam = lam_array[mid_index]
	down = lam_array[:mid_index-1]
	up = lam_array[mid_index+1:]

	v_i_list = list(itertools.product(nodes, cascades))
	
	v_i_len = len(v_i_list)
	pool_len = len(pools)
	casc_len = len(cascades)

	#define the vectors c and b as defined in the dual program
	c_vec = np.array([1 if x not in casc else tau for (x, casc) in cascades]) #[1] * len(v_i_list))
	b_vec = np.array([lam/casc_len] * pool_len + [1/casc_len] * v_i_len)


	delta = (1 + epsilon) *  ((1+epsilon) * (v_i_len + pool_len)) ** (-1/epsilon)

	y_0 = delta / b_vec
	y = y_0
	primal = 0

	itera = 0

	while np.dot(b_vec, y) <= 1:
		# Do the iterations here
		length_vector = (A.T @ y) / c_vec # (A.T * np.atleast_2d(y)) # This is incorrect - Need to fix this definition
		#print(f'length vector shape: {length_vector.shape}')

		alpha_y = np.min(length_vector) ; q = np.argmin(length_vector) ;
		#print(f'q: {q}, A shape: {A.shape}')
		min_capacity_edge_vec = b_vec / A[:, q]
		#print(min_capacity_edge_vec)

		min_capacity_edge = np.min(min_capacity_edge_vec) ; p = np.argmin(min_capacity_edge) ;

		#update of primal LP
		primal += (b_vec[p]*c_vec[p]) / A[p,q]

		#update of dual (our LP)
		y = y * (1 + epsilon * (b_vec[p] / A[p,q]) / (b_vec / np.squeeze(A[:, q])))
		itera += 1
		#print(f'Current Iteration: {itera}')

	#break up the vector here, the final elements are the pools, the earlier elements are the node/cascade tuples
	x_s = np.array(y[:pool_len])
	z_i_d = np.array(y[-v_i_len:])

	# z = 1 - y, which means y = 1-z
	print(f'Objective Value: {np.dot(b_vec, y)}')
	return x_s, z_i_d

# current_full_y is a vector of the both the vector of cleared values and a sum with the pools
# Recall 1-y = z -> y = z+1



	
def acceptable_range(budget, sets_output, lam, eta=.05):
	if np.absolute((np.sum(sets_output) - budget)) < budget * eta:
		return 0, lam
	elif np.sum(sets_output) > budget:
		return -1, lam
	else:
		return 1, lam


def binary_search(mini, maxi, sets_output, budget):
	val = (mini+maxi) / 2
	bool_val, lam = acceptable_range(budget, sets_output, val)
	if bool_val == 0:
		
		return True, 0, 0
	elif bool_val == -1:
		#return False, array[:mid_index+1:]
		return False, val, maxi
	else:
		#return False, array[:mid_index-1]
		return False, mini, val

def read_graph(name):
	if name == 'test_graph':
		G = nx.read_edgelist('data/test_graph.txt')
	elif name == 'test_grid':
		G = nx.read_edgelist('data/test_grid.txt')
	elif name == 'lyon':
		df = pd.read_csv('data/hospital_contacts', sep='\t', header=None)
		df.columns = ['time', 'e1', 'e2', 'lab_1', 'lab_2']
		G = nx.from_pandas_edgelist(df, 'e1', 'e2')
	elif name == 'bird':
		G = nx.read_edgelist('data/aves-wildbird-network.edges')
	elif name == 'tortoise':
		G = nx.read_edgelist('data/reptilia-tortoise-network-fi-2011.edges')
	elif name == 'dolphin':
		G = nx.read_edgelist('data/mammalia-dolphin-florida-overall.edges')
	elif name == 'uva_pre':
		network = open('data/personnetwork_exp', 'r')
		lines = network.readlines()
		lst = []
		for line in lines:
			lst.append(line.strip())
		network.close()
		H_prime = nx.parse_edgelist(lst[:450])
		G = H_prime.subgraph(max(nx.connected_components(H_prime))).copy()
		del lst
	elif name == 'uva_post':
		network = open('data/personnetwork_post', 'r')
		lines = network.readlines()
		lst = []
		for line in lines:
			lst.append(line.strip())
		network.close()
		H_prime = nx.parse_edgelist(lst[1000:1500])
		G = H_prime.subgraph(max(nx.connected_components(H_prime))).copy()
	elif name == 'random':
		G = nx.read_edgelist('data/random_150_0.08_12.txt')

	mapping = dict(zip(G.nodes(),range(len(G))))
	graph = nx.relabel_nodes(G,mapping)
	return graph


if __name__ == "__main__":
	start_time = time.time()
	#So this graph is only 75 nodes, 1138 edges
	print('Here we go!')

	graph = read_graph('lyon')
	
	#set_list = enumerate(graph)
	set_list = enumerate_random(graph, 5)

	print(f'Pools enumerated: {time.time() - start_time} seconds ---')
	cascade_list = cascade_construction(graph, 250, .1)

	# Doing it this way because A only has to be constructed once and is a major time suck
	# We have to do the approximation a few times to settle on lambda
	A = define_A_matrix(set_list, list(graph.nodes()), cascade_list)
	print(f'A Matrix Construction: {time.time() - start_time} seconds ---')

	done = False
	#x_s, y_i_d = approximation(A, set_list, list(graph.nodes()), cascade_list, lam=(mini+maxi)/2, epsilon=.1)
	mini = 1/len(graph) ; maxi = len(graph) ** 2
	budget = 5 #int(np.log(len(graph.nodes())))

	it = 0

	while not done:
		lam = (mini+maxi) / 2
		x_s, z_i_d = approximation(A, set_list, list(graph.nodes()), cascade_list, lam=lam, epsilon=.05)
		print(f'Lambda Guess: {lam}, number of sets: {sum(x_s)}')
		done, mini, maxi = binary_search(mini, maxi, x_s, budget)
		if it > 5000:
			print(it)
			done = True
		it += 1

	#x_s, y_i_d = approximation(set_list, list(graph.nodes()), cascade_list, epsilon=.1)

	#x_prime = rounding(x)

	#rounded_obj_val = calculate_E_welfare(x_prime, cascade_list)
	# z = 1 - y, which means y = 1-z
	np.set_printoptions(threshold=sys.maxsize)
	#print(f'Any OOB? {not all((z_i_d > 0) & (z_i_d < 1))}')
	print(f'Maximum Value: {max(z_i_d)}')
	#print(f'Array: {z_i_d}')
	print(f'Number of sets chosen: {np.sum(x_s)}, Expected Welfare: {np.sum([1 - x for x in z_i_d]) * 1/len(cascade_list)}')
	print(f'Total Run Time: {time.time() - start_time} seconds ---')
	#print(f'LP Obj Val: {obj_value}, Rounded Obj Val: {rounded_obj_val}, size of x, y: {len(x)}, {len(y)}')




