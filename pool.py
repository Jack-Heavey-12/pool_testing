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


from collections import defaultdict
import itertools

from matplotlib import pyplot as plt

def enumerate(graph, n_p=5):
	#returns list of the possible subsets that can be chosen
	#NOTE: This list is O(n^{n_p}), be aware of memory constraints
	nodes = list(graph.nodes())
	set_list = []
	for i in range(1, n_p+1):
		set_list += list(itertools.combinations(nodes, i))
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
	return cascade_list



'''REQUIRED DATA TYPES FOR INPUTS:
	- graph: networkx.Graph()
	- set_list: list of tuples, possible available sets
	- cascades: A list of lists of connected components - they correspond directly to the nodes in the graph
	- B: int
	- overlapping: bool (defines whether overlapping sets are allowed or not, affects constraint 1)
'''
def LinearProgram(graph, set_list, cascades, B, overlapping=True):

	x = {} # defined in the paper
	y = {} # defined in the paper

	N = len(cascades)

	m = Model('pool_testing')

	for S in set_list:
		x[S] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = f'x[{S}]')

	node_list = list(graph.nodes())
	for i in range(len(cascades)):
		for v in node_list:
			y[f'({v}, {i})'] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = f'y[({v}, {i})]')

	m.update()

	# Constraint 1
	#If we are doing overlapping sets
	if overlapping:
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
	#Non-overlapping Constraint
	#END CONSTRAINT ONE

	m.update()
	

	# Constraint 2 - need results to be under the budget
	m.addConstr(quicksum(x[S] for S in set_list) <= B, name='C2_Budget Constraint')

	m.update()


	m.setObjective(1/N * quicksum(y[f'({v}, {i})'] for i in range(N) for v in node_list), GRB.MAXIMIZE)
	m.update()
	m.optimize()

	#RETURNS THE DICTIONARY WITH THE VARIABLES X (FOR ROUNDING LATER), DICTIONARY WITH VARIABLES Y, 
	#		AND THE OPTIMAL OBJECTIVE VALUE (UPPER BOUND ON ROUNDED ANSWER WITH NO VIOLATED BUDGET)
	return x, y, m.objVal


def rounding(x_dict):
	#Will return the rounded values in a dictionary, which correspond to specific subsets to choose
	#Doing this in a separate function in case the rounding function changes later

	x_prime_dict = {}
	for S in x.keys():
		limit = np.random.uniform(0,1)
		if limit <= x[S].x:
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
				val = int(all([False if v in cc else True for cc in i]))
				running_clearances += val
	return running_clearances / num_cascades #The correct objective value I think?

	#Cycle through keys in x_prime, see which cascades they are cleared in

if __name__ == "__main__":

	#So this graph is only 75 nodes, 1138 edges.
	df = pd.read_csv('data/hospital_contacts', sep='\t', header=None)
	df.columns = ['time', 'e1', 'e2', 'lab_1', 'lab_2']
	graph = nx.from_pandas_edgelist(df, 'e1', 'e2')
	#graph = nx.read_edgelist(INSERT FILE NAME HERE)
	
	set_list = enumerate(graph)
	cascade_list = cascade_construction(graph, 1000, .05)

	x, y, obj_value = LinearProgram(graph, set_list, cascade_list, 10)

	x_prime = rounding(x)

	rounded_obj_val = calculate_E_welfare(x_prime, cascade_list)

	print(f'LP Obj Val: {obj_value}, Rounded Obj Val: {rounded_obj_val}, size of x, y: {len(x)}, {len(y)}')



