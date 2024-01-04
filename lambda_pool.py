# File that defines the algorithm for pool testing

import networkx as nx
import sys
from gurobipy import *
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

'''def acceptable_range(budget, sets_output, lam, eta=.05):
	if np.absolute((np.sum(sets_output) - budget)) < eta: #budget * eta:
		return 0, lam
	elif np.sum(sets_output) > budget:
		return -1, lam
	else:
		return 1, lam
'''

def binary_search(mini, maxi, x_dict, budget, eta=.25):
	su = np.sum(list(x_dict.values()))
	if np.abs((su - budget)/budget) <= .1:
		return True, 0, 0
	#if lam - mini <.05:
	#	return False, lam/2, lam*2
	if su > budget:
		return False, lam, maxi
	else:
		return False, mini, lam

def best_guess(x, budget, prior_best):
	su = np.sum(list(x.values())); p_su = np.sum(list(prior_best.values())); 
	if np.abs(su - budget) < np.abs(p_su - budget):
		return x
	else:
		return prior_best




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
	return cascade_list



'''REQUIRED DATA TYPES FOR INPUTS:
	- graph: networkx.Graph()
	- set_list: list of tuples, possible available sets
	- cascades: A list of lists of connected components - they correspond directly to the nodes in the graph
	- B: int
	- overlapping: bool (defines whether overlapping sets are allowed or not, affects constraint 1)
'''
def LinearProgram(graph, set_list, cascades, B=3, lam=1.01, overlapping=True):

	x = {} # defined in the paper
	z = {} # defined in the paper

	N = len(cascades)

	m = Model('pool_testing')

	for S in set_list:
		x[S] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = f'x[{S}]')

	node_list = list(graph.nodes())
	for i in range(len(cascades)):
		for v in node_list:
			z[f'(node_{v}, casc_{i})'] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = f'z[({v}, {i})]')

	m.update()

	# Constraint 1
	#If we are doing overlapping sets
	'''if overlapping:
		for i in range(len(cascades)):
			for v in node_list:
				#print(cascades[i])
				val =  any([x if v in x else False for x in cascades[i]]) #Boolean for if the node is in ANY connected conponent from any source
				if not val:
					# The node is not in the connected component, any set with this node in it is valid:
					F_vi = [S for S in set_list if v in S]
					m.addConstr(y[f'({v}, {i})'] <= quicksum(x[S] for S in F_vi), name=f'C1_Node_{v}_cascade_{i}_overlapping')
	#If we have non-overlapping sets, the same thing but with equality in the constraint rather than an inequality	
	else:
		for i in range(len(cascades)):
			for v in node_list:
				val =  any([x if v in x else False for x in cascades[i]]) #Boolean for if the node is in ANY connected conponent from any source
				if not val:
					# The node is not in the connected component, any set with this node in it is valid:
					F_vi = [S for S in set_list if v in S]
					m.addConstr(y[f'({v}, {i})'] == quicksum(x[S] for S in F_vi), name=f'C1_Node_{v}_cascade_{i}')
	#Non-overlapping Constraint'''

	for i in range(len(cascades)):
			for v in node_list:
				val =  v in cascades[i] #Boolean for if the node is in the connected conponent from the source
				if not val:
					# The node is not in the connected component, any set with this node in it is valid:
					F_vi = [S for S in set_list if v in S]
					m.addConstr(quicksum(x[S] for S in F_vi) + z[f'(node_{v}, casc_{i})'] >= 1, name=f'C1_Node_{v}_cascade_{i}')
	#END CONSTRAINT ONE

	m.update()
	

	# Constraint 2 - need results to be under the budget
	#m.addConstr(quicksum(x[S] for S in set_list) <= B, name='C2_Budget Constraint')

	m.update()

	possible_cleared_nodes = []
	for i in range(len(cascades)):
		for v in node_list:
			val =  v in cascades[i]
			if not val:
				possible_cleared_nodes.append((v,i))


	m.setObjective(1/N * quicksum(z[f'(node_{v}, casc_{i})'] for (v,i) in possible_cleared_nodes) + lam * quicksum(x[S] for S in set_list),
	 				GRB.MINIMIZE)

	m.setParam('OutputFlag', 1)
	m.update()
	m.optimize()
	print(f'Status Code: {m.Status}')

	print(f'Solution Count: {m.SolCount}')


	#RETURNS THE DICTIONARY WITH THE VARIABLES X (FOR ROUNDING LATER), DICTIONARY WITH VARIABLES Y, 
	#		AND THE OPTIMAL OBJECTIVE VALUE (UPPER BOUND ON ROUNDED ANSWER WITH NO VIOLATED BUDGET)
	variables = m.getVars()
	x_vals = {}
	z_vals = {}
	for i in x.keys():
		x_vals[i] = x[i].X
	for i in z.keys():
		z_vals[i] = z[i].X

	return x_vals, z_vals, m.objVal, variables


def rounding(x_dict):
	#Will return the rounded values in a dictionary, which correspond to specific subsets to choose
	#Doing this in a separate function in case the rounding function changes later

	x_prime_dict = {}
	for S in x.keys():
		limit = np.random.uniform(0,1)
		#print(type(x[S]))
		#print(dir(x[S]))
		#sys.exit()
		if limit <= x[S]:
			x_prime_dict[S] = 1
	#NOTE: 	This will not have *every* set in this dictionary, only the pools that we are going to end up choosing
	#		Figure this is easier than trying to check if every value is one or zero, we can jsut compare length for expectation, etc.

	#NOTE2:	The keys in x, and thus x_prime_dict, are the sets themselves. Think that's the best way to do it, shouldn't have any assignment issues?
	return x_prime_dict


def calculate_E_welfare(x_prime, cascade_list):
	#TODO
	num_cascades = len(cascade_list)
	running_clearances = 0
	for S in x_prime.keys():
		#x_prime.keys() is going to be a set, which means have to iterate through the set and then each cascade to see if it's cleared.
		for v in S:
			for i in cascade_list:
				#will give a 1 if the node is cleared in that cascade, note we DON'T want it in the connected component
				val = int(False if v in i else True) * x_prime[S] #will multiply by the fractional value.
				running_clearances += val
	return running_clearances / num_cascades #The correct objective value I think?

	#Cycle through keys in x_prime, see which cascades they are cleared in

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
	elif name == 'path_graph':
		G = nx.read_edgelist('data/path_graph_4.txt')

	mapping = dict(zip(G.nodes(),range(len(G))))
	graph = nx.relabel_nodes(G,mapping)
	return graph

if __name__ == "__main__":

	graph = read_graph('test_graph')
	#graph = read_graph('path_graph')

	
	set_list = enumerate(graph, n_p=3)
	#set_list = enumerate_random(graph, n_p=3, num_sets=1000)
	#cascade_list = cascade_construction(graph, 1000, .05)
	with open('test_cascades/test_graph_100_0.1.pkl', 'rb') as f:
	#with open('test_cascades/path_graph_4n_5c.pkl', 'rb') as f:
	#with open('test_cascades/uva_pre_1000_0.5.pkl', 'rb') as f:
		cascade_list = pickle.load(f)

	fl = .1

	done = False
	#x_s, y_i_d = approximation(A, set_list, list(graph.nodes()), cascade_list, lam=(mini+maxi)/2, epsilon=.1)
	mini = 1/len(graph) ** 2; maxi = (len(cascade_list) * len(graph)) ** 2
	budget = 5 #int(np.log(len(graph.nodes())))
	lam = len(graph)
	convex_ep=.5
	it = 0
	prior_best = {}


	while not done:
		#lam = (mini+maxi) / 2
		x, z, obj_value, variables = LinearProgram(graph, set_list, cascade_list, budget, lam=lam)

		print(f'Lambda Guess: {lam}')
		#print(f'X Dict: {x}')
		done, mini, maxi = binary_search(mini, maxi, x, budget, eta=.05)
		if it > 5000:
			print(it)
			done = True
		it += 1
		prior_best = best_guess(x, budget, prior_best)
		lam = mini * convex_ep + (1-convex_ep) * maxi
		#print(f'Variables: {x}, {z}')
	
	expected_welfare = 0
	for i in z.keys():
		expected_welfare += 1-z[i]
	#print(f'Expected Welfare: {expected_welfare/len(cascade_list)}')

	#print(f'\n\nCascades: {[i.nodes() for i in cascade_list]}\n\n')

	#print(f'Variables: {variables}')

	x = prior_best

	x_prime = rounding(x)
	nonzeros = {}
	for i in x.keys():
		if x[i] > 0:
			nonzeros[i] = x[i]
	#print(f'\n\nCascades: {[i.nodes() for i in cascade_list]}\n\n')
	#print(f'LP Output: {nonzeros}, Ys: {y}')
	print(f'Sets Chosen: {nonzeros}')

	pre_rounded_obj_val = calculate_E_welfare(nonzeros, cascade_list)
	rounded_obj_val = calculate_E_welfare(x_prime, cascade_list)

	with open('lam_results.csv', 'a') as f:
		f.write(f'\n{len(graph)},{len(cascade_list)},{fl},{pre_rounded_obj_val},{rounded_obj_val}, Y, {len(set_list)}, 3,{budget}')

	print(f'Expected Welfare: {pre_rounded_obj_val}, Actual Welfare from Cascades: {rounded_obj_val}, size of x, y: {len(x)}, {len(z)}')



