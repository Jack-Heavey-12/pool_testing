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

from matplotlib import pyplot as plt

import time


# NOTE: This will return A as defined, NOT A^T, which is what is defined by our LP
def define_a_matrix(pools, nodes, cascades, tau=1e-10):
	v_i_list = list(itertools.product(nodes, cascades))

	v_i_len = len(v_i_list)
	pool_len = len(pools)
	casc_len = len(cascades)

	#size of matrix is going to be pool_len + v_i_len rows, v_i_len for columns
	A = np.zeros((v_i_len + pool_len, v_i_len))

	for i in range(pool_len):
		# This is taking into account note 2 for Anil, where it is checking to see if any of the values in the set fall in the cascade (which is when it would result in a negative test)
		A[i + v_i_len] = np.array([1 if ((x in pools[i]) and not any(x in casc for x in pools[i])) else tau for (x, casc) in v_i_list])
		#A[i, :] = np.array([1 if (x in pools[i]) else tau for (x, casc) in v_i_list])

	A_vid = np.identity(v_i_len)
	inds = A_vid == 0
	A_vid[inds] = tau

	for i in range(v_i_len):
		A[i] = A_vid[i, :]

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

	v_i_list = list(itertools.product(nodes, cascades))
	
	v_i_len = len(v_i_list)
	pool_len = len(pools)
	casc_len = len(cascades)

	#define the vectors c and b as defined in the dual program
	c_vec = np.array([1 if x not in casc else tau for (x, casc) in v_i_list]) #[1] * len(v_i_list))
	b_vec = np.array([1/casc_len] * v_i_len + [lam] * pool_len)


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
	z_i_d = np.array(y[:v_i_len])
	x_s = np.array(y[-pool_len:])

	#x_s = np.array(y[:pool_len])
	#z_i_d = np.array(y[-v_i_len:])

	# z = 1 - y, which means y = 1-z
	print(f'Objective Value: {np.dot(b_vec, y)}')
	return x_s, z_i_d

# current_full_y is a vector of the both the vector of cleared values and a sum with the pools
# Recall 1-y = z -> y = z+1


def binary_search(mini, maxi, x_dict, budget, eta=.25):
	su = np.sum(x_dict)
	if np.abs((su - budget)/budget) <= .1:
		return True, 0, 0
	#if lam - mini <.05:
	#	return False, lam/2, lam*2
	if su > budget:
		return False, lam, maxi
	else:
		return False, mini, lam

def best_guess(x, budget, prior_best):
	su = np.sum(x); p_su = np.sum(prior_best); 
	if np.abs(su - budget) < np.abs(p_su - budget):
		return x
	else:
		return prior_best




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
		H_prime = nx.parse_edgelist(lst[:6000])
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

	graph = read_graph('test_graph')
	
	set_list = enumerate(graph, n_p=3)
	#set_list = enumerate_random(graph, 5)

	#with open('test_cascades/test_graph_100_0.1.pkl', 'rb') as f:
	#with open('test_cascades/path_graph_4n_5c.pkl', 'rb') as f:
	with open('test_cascades/uva_pre_1000_0.33.pkl', 'rb') as f:
		cascade_list = pickle.load(f)

	# Doing it this way because A only has to be constructed once and is a major time suck
	# We have to do the approximation a few times to settle on lambda
	A = define_A_matrix(set_list, list(graph.nodes()), cascade_list)
	print(f'A Matrix Construction: {time.time() - start_time} seconds ---')

	done = False
	#x_s, y_i_d = approximation(A, set_list, list(graph.nodes()), cascade_list, lam=(mini+maxi)/2, epsilon=.1)
	mini = 1/len(graph) ** 2; maxi = (len(cascade_list) * len(graph)) ** 2
	budget = 5 #int(np.log(len(graph.nodes())))

	lam = len(graph)
	convex_ep=.5
	it = 0
	prior_best = []

	it = 0

	'''while not done:
		lam = (mini+maxi) / 2
		x_s, z_i_d = approximation(A, set_list, list(graph.nodes()), cascade_list, lam=lam, epsilon=.05)
		print(f'Lambda Guess: {lam}, number of sets: {sum(x_s)}')
		done, mini, maxi = binary_search(mini, maxi, x_s, budget, convex_ep=.2)
		if it > 5000:
			print(it)
			done = True
		it += 1'''

	while not done:
		#lam = (mini+maxi) / 2
		x_s, z_i_d = approximation(A, set_list, list(graph.nodes()), cascade_list, lam=lam, epsilon=.05)
		print(f'Lambda Guess: {lam}, number of sets: {sum(x_s)}')
		done, mini, maxi = binary_search(mini, maxi, x_s, budget)
		#print(f'X Dict: {x}')
		if it > 5000:
			print(it)
			done = True
		it += 1
		prior_best = best_guess(x_s, budget, prior_best)
		lam = mini * convex_ep + (1-convex_ep) * maxi
		#print(f'Variables: {x}, {z}')



	#x_s, y_i_d = approximation(set_list, list(graph.nodes()), cascade_list, epsilon=.1)

	#x_prime = rounding(x)

	#rounded_obj_val = calculate_E_welfare(x_prime, cascade_list)
	# z = 1 - y, which means y = 1-z
	#np.set_printoptions(threshold=sys.maxsize)
	#print(f'Any OOB? {not all((z_i_d > 0) & (z_i_d < 1))}')
	print(f'Maximum Value: {max(z_i_d)}')
	#print(f'Array: {z_i_d}')
	print(f'Number of sets chosen: {np.sum(x_s)}, Expected Welfare: {np.sum([1 - x for x in z_i_d]) * 1/len(cascade_list)}')
	print(f'Total Run Time: {time.time() - start_time} seconds ---')
	#print(f'LP Obj Val: {obj_value}, Rounded Obj Val: {rounded_obj_val}, size of x, y: {len(x)}, {len(y)}')

