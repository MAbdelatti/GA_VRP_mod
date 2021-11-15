#!/usr/bin/env python
# -------- Start of the importing part -----------
from numba import cuda, jit, int32, float32, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import cupy as cp
from math import pow, hypot, ceil, floor, log
from timeit import default_timer as timer
import numpy as np
import random
import sys
from datetime import datetime
import gpuGrid
import val
import time
# -------- End of the importing part -----------
np.set_printoptions(threshold=sys.maxsize)

# ------------------------- Start reading the data file -------------------------------------------
class vrp():
    def __init__(self, capacity=0, opt=0):
        self.capacity = capacity
        self.opt = opt
        self.nodes = np.zeros((1,4), dtype=np.float32)
    def addNode(self, label, demand, posX, posY):
        newrow = np.array([label, demand, posX, posY], dtype=np.float32)
        self.nodes = np.vstack((self.nodes, newrow))

def readInput():

    # Create VRP object:
    vrpManager = vrp()
    # First reading the VRP from the input #
    print('Reading data file...', end=' ')
    fo = open(sys.argv[1],"r")
    lines = fo.readlines()
    for i, line in enumerate(lines):       
        while line.upper().startswith('COMMENT'):
            if len(sys.argv) <= 3:
                inputs = line.split()
                if inputs[-1][:-1].isnumeric():
                    vrpManager.opt = np.int32(inputs[-1][:-1])
                    break
                else:
                    try:
                        vrpManager.opt = float(inputs[-1][:-1])
                    except:
                        print('\nNo optimal value detected, taking optimal as 0.0')
                        vrpManager.opt = 0.0
                    break
            else:
                vrpManager.opt = np.int32(sys.argv[3])
                print('\nManual optimal value entered: %d'%vrpManager.opt)
                break

        # Validating positive non-zero capacity
        if vrpManager.opt < 0:
            print(sys.stderr, "Invalid input: optimal value can't be negative!")
            exit(1)
            break

        while line.upper().startswith('CAPACITY'):
            inputs = line.split()
            try:
                vrpManager.capacity = np.float32(inputs[2])
            except IndexError:
                vrpManager.capacity = np.float32(inputs[1])
			# Validating positive non-zero capacity
            if vrpManager.capacity <= 0:
                print(sys.stderr, 'Invalid input: capacity must be neither negative nor zero!')
                exit(1)
            break       
        while line.upper().startswith('NODE_COORD_SECTION'):
            i += 1
            line = lines[i]
            while not (line.upper().startswith('DEMAND_SECTION') or line=='\n'):
                inputs = line.split()
                vrpManager.addNode(np.int16(inputs[0]), 0.0, np.float32(inputs[1]), np.float32((inputs[2])))

                i += 1
                line = lines[i]
                while (line=='\n'):
                    i += 1
                    line = lines[i]
                    if line.upper().startswith('DEMAND_SECTION'): break 
                if line.upper().startswith('DEMAND_SECTION'):
                    i += 1
                    line = lines[i] 
                    while not (line.upper().startswith('DEPOT_SECTION')):                  
                        inputs = line.split()
						# Validating demand not greater than capacity
                        if float(inputs[1]) > vrpManager.capacity:
                            print(sys.stderr,
							'Invalid input: the demand of the node %s is greater than the vehicle capacity!' % vrpManager.nodes[0])
                            exit(1)
                        if float(inputs[1]) < 0:
                            print(sys.stderr,
                            'Invalid input: the demand of the node %s cannot be negative!' % vrpManager.nodes[0])
                            exit(1)                            
                        vrpManager.nodes[int(inputs[0])][1] =  float(inputs[1])
                        i += 1
                        line = lines[i]
                        while (line=='\n'):
                            i += 1
                            line = lines[i]
                            if line.upper().startswith('DEPOT_SECTION'): break
                        if line.upper().startswith('DEPOT_SECTION'):
                            vrpManager.nodes = np.delete(vrpManager.nodes, 0, 0) 
                            print('Done.')
                            return(vrpManager.capacity, vrpManager.nodes, vrpManager.opt)

# ------------------------- Calculating the cost table --------------------------------------
@cuda.jit
def calculateLinearizedCost(data_d, linear_cost_table):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)
    
    for row in range(threadId_row, data_d.shape[0], stride_x):
        for col in range(threadId_col, data_d.shape[0], stride_y):
            if col > row:
                k = int(col - (row*(0.5*row - data_d.shape[0] + 1.5)) - 1)
                linear_cost_table[k] = \
                round(hypot(data_d[row, 2] - data_d[col, 2], data_d[row, 3] - data_d[col, 3]))

# ------------------------- Fitness calculation ---------------------------------------------
@cuda.jit
def fitness_gpu_old(linear_cost_table, pop, n):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        fitnessValue = 0
        pop[row, -1] = 1
        
        if threadId_col == 15:
            for idx in range(1, pop.shape[1]-2):
                i = min(pop[row, idx]-1, pop[row, idx+1]-1)
                j = max(pop[row, idx]-1, pop[row, idx+1]-1)

                if i != j:
                    k = int(j - (i*(0.5*i - n + 1.5)) - 1)
                    fitnessValue += linear_cost_table[k]

            # bit_count     = int((log(fitnessValue) /  log(2)) + 1)
            scaledFitness = fitnessValue  # Scaling the fitness to fit int16
            # scaledFitness = fitnessValue >> bit_count - 16 # Scaling the fitness to fit int16
            pop[row, -1]  = scaledFitness
    
    cuda.syncthreads()

@cuda.jit
def computeFitness(linear_cost_table, pop, n):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1]-2, stride_y):
            i = min(pop[row, col]-1, pop[row, col+1]-1)
            j = max(pop[row, col]-1, pop[row, col+1]-1)

            if i != j:
                k = int(j - (i*(0.5*i - n + 1.5)) - 1)

                cuda.atomic.add(pop, (row, pop.shape[1]-1), linear_cost_table[k])
   
# ------------------------- Refining solutions ---------------------------------------------
@cuda.jit
def find_duplicates_old(pop, r_flag):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:
            # Detect duplicate nodes:
            for i in range(2, pop.shape[1]-1):
                for j in range(i, pop.shape[1]-1):
                    if pop[row, i] != r_flag and pop[row, j] == pop[row, i] and i != j:
                        pop[row, j] = r_flag

@cuda.jit
def find_missing_nodes_old(data_d, missing_d, pop):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        if threadId_col == 15:
            missing_d[row, threadId_col] = 0        
            # Find missing nodes in the solutions:
            for i in range(1, data_d.shape[0]):
                for j in range(2, pop.shape[1]-1):
                    if data_d[i,0] == pop[row,j]:
                        missing_d[row, i] = 0
                        break
                    else:
                        missing_d[row, i] = data_d[i,0]                  

@cuda.jit
def add_missing_nodes_old(missing_d, pop, r_flag):   
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):       
        if threadId_col == 15:           
            # Add the missing nodes to the solution:
            for k in range(missing_d.shape[1]):
                for l in range(2, pop.shape[1]-1):
                    if missing_d[row, k] != 0 and pop[row, l] == r_flag:
                        pop[row, l] = missing_d[row, k]
                        break                        

@cuda.jit
def find_duplicates(pop, r_flag):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1]-1, stride_y): 
            if col >= 2:
                for j in range(col+1, pop.shape[1]-1):
                    if pop[row, col] != r_flag and pop[row, j] == pop[row, col]:
                        pop[row, j] = r_flag                        

@cuda.jit
def prepareAuxiliary(data_d, missing_d):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, missing_d.shape[0], stride_x):
        for col in range(threadId_col, missing_d.shape[1], stride_y):
            if col < data_d.shape[0]:
                missing_d[row, col] = 0

@cuda.jit
def findExistingNodes(n, pop, exitsing_nodes):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1], stride_y):            
            if pop[row, col] <= n:
                exitsing_nodes[row, pop[row, col]-1] = pop[row, col]            
            
@cuda.jit
def deriveMissingNodes(data_d, missing_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, missing_d.shape[0], stride_x):
        for col in range(threadId_col, missing_d.shape[1], stride_y):
            if col < data_d.shape[0]:
                if missing_d[row, col] == 0:
                    missing_d[row, col] = col+1
                else:
                    missing_d[row, col] = 0                

@cuda.jit
def addMissingNodes(r_flag, missing_d, pop):   
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, missing_d.shape[1], stride_y):
            if missing_d[row, col] != 0:
                for j in range(2, pop.shape[1]-1):
                    cuda.atomic.compare_and_swap(pop[row,j:], r_flag, missing_d[row, col])
                    if pop[row, j] == missing_d[row, col]:
                        break

@cuda.jit
def shift_r_flag(r_flag, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:           
            # Shift all r_flag values to the end of the list:        
            for i in range(2, pop.shape[1]-2):
                if pop[row,i] == r_flag:
                    k = i
                    while pop[row,k] == r_flag:
                        k += 1
                    if k < pop.shape[1]-1:
                        pop[row,i], pop[row,k] = pop[row,k], pop[row,i]

@cuda.jit
def cap_adjust(r_flag, vrp_capacity, data_d, pop):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        if threadId_col == 15:
            reqcap = 0.0        # required capacity
            
            # Accumulate capacity:
            i = 1
            while pop[row, i] != r_flag:
                i += 1  
                if pop[row,i] == r_flag:
                    break
            
                if pop[row, i] != 1:
                    reqcap += data_d[pop[row, i]-1, 1] # index starts from 0 while individuals start from 1                
                    if reqcap > vrp_capacity:
                        reqcap = 0
                        # Insert '1' and shift right:
                        new_val = 1
                        rep_val = pop[row, i]
                        for j in range(i, pop.shape[1]-2):
                            pop[row, j] = new_val
                            new_val = rep_val
                            rep_val = pop[row, j+1]
                else:
                    reqcap = 0.0
    cuda.syncthreads()

@cuda.jit
def cleanup_r_flag(r_flag, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1], stride_y):
            if pop[row, col] == r_flag:
                pop[row, col] = 1
    
# ------------------------- Start initializing individuals ----------------------------------------
@cuda.jit
def initializePop(data_d, pop_d):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)
    
    # Generate the individuals from the nodes in data_d:
    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, data_d.shape[0]+1, stride_y):
            pop_d[row, col] = data_d[col-1, 0]
        
        pop_d[row, 0], pop_d[row, 1] = 1, 1
        
# ------------------------- Start two-opt calculations --------------------------------------------
@cuda.jit
def reset_to_ones(pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        for col in range(threadId_col, pop.shape[1], stride_y):
            pop[row, col] = 1   

@cuda.jit
def twoOpt(pop, auxiliary_arr, cost_table, n):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        for col in range(threadId_col, pop.shape[1], stride_y):
            if col+2 < pop.shape[1]:
                # Divide solution into routes, active threads are the firs 1's in the route:
                if pop[row, col] == 1 and pop[row, col+1] != 1 and pop[row, col+2] != 1:
                    route_length = 1
                    while pop[row, col+route_length] != 1 and col+route_length < pop.shape[1]:
                        auxiliary_arr[row, col+route_length] = pop[row, col+route_length]
                        route_length += 1

                    # Now we have auxiliary_arr has the routes to be optimized for every row solution
                    total_cost = 0
                    min_cost =0

                    # compute the route cost:
                    for idx in range(0, route_length):
                        i = min(auxiliary_arr[row,col+idx]-1, auxiliary_arr[row,col+idx+1]-1)
                        j = max(auxiliary_arr[row,col+idx]-1, auxiliary_arr[row,col+idx+1]-1)

                        if i != j:
                            k = int(j - (i*(0.5*i - n + 1.5)) - 1)
                            min_cost += cost_table[k]
                           
                    # So far, the best route is the given one (in auxiliary_arr)
                    improved = True
                    while improved:
                        improved = False
                        for idx_i in range(1, route_length-1):
                                # swap every two pairs
                                auxiliary_arr[row, col+idx_i]  , auxiliary_arr[row, col+idx_i+1] = \
                                auxiliary_arr[row, col+idx_i+1], auxiliary_arr[row, col+idx_i]

                                # check the cost of the arrangement:
                                for idx_j in range(0, route_length):
                                    i = min(auxiliary_arr[row,col+idx_j]-1, auxiliary_arr[row,col+idx_j+1]-1)
                                    j = max(auxiliary_arr[row,col+idx_j]-1, auxiliary_arr[row,col+idx_j+1]-1)

                                    if i != j:
                                        k = int(j - (i*(0.5*i - n + 1.5)) - 1)
                                        total_cost += cost_table[k]
                                
                                if total_cost < min_cost:
                                    min_cost = total_cost
                                    improved = True
                                else:
                                    auxiliary_arr[row, col+idx_i+1], auxiliary_arr[row, col+idx_i]=\
                                    auxiliary_arr[row, col+idx_i]  , auxiliary_arr[row, col+idx_i+1]
                    
                    for idx_k in range(0, route_length):
                        pop[row, col+idx_k] = auxiliary_arr[row, col+idx_k]                        

# --------------------------------- Cross Over part ---------------------------------------------
@cuda.jit  
def selectParents(pop_d, random_arr_d, parent_idx):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, pop_d.shape[1], stride_y):  
            candid_1_idx = random_arr_d[row, 0]
            candid_2_idx = random_arr_d[row, 1]
            candid_3_idx = random_arr_d[row, 2]
            candid_4_idx = random_arr_d[row, 3]

            # Selecting 1st Parent from binary tournament:
            if pop_d[candid_1_idx, -1] < pop_d[candid_2_idx, -1]:
                parent_idx[row, 0] = candid_1_idx
            else:
                parent_idx[row, 0] = candid_2_idx

            # Selecting 2nd Parent from binary tournament:
            if pop_d[candid_3_idx, -1] < pop_d[candid_4_idx, -1]:
                parent_idx[row, 1] = candid_3_idx
            else:
                parent_idx[row, 1] = candid_4_idx
       
@cuda.jit
def getParentLengths(no_of_cuts, pop_d, auxiliary_arr, parent_idx):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, auxiliary_arr.shape[0], stride_x):
        for col in range(threadId_col, auxiliary_arr.shape[1], stride_y):
            auxiliary_arr[row, col] = 1

        # Calculate the actual length of parents
        if col < pop_d.shape[1]-2:
            if not (pop_d[parent_idx[row, 0], col] == 1 and pop_d[parent_idx[row, 0], col+1] == 1):
                cuda.atomic.add(auxiliary_arr, (row, 2), 1)
                
            if not (pop_d[parent_idx[row, 1], col] == 1 and pop_d[parent_idx[row, 1], col+1] == 1):
                cuda.atomic.add(auxiliary_arr, (row, 3), 1)

            # Minimum length of the two parents
            cuda.atomic.min(auxiliary_arr, (row, 3), auxiliary_arr[row, 2])

            auxiliary_arr[row, 4] = no_of_cuts # k-point crossover

@cuda.jit
def add_cut_points(auxiliary_arr, rng_states):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, auxiliary_arr.shape[0], stride_x):
        if threadId_col == 15:
            no_cuts = auxiliary_arr[row, 4]
            
            for i in range(5, no_cuts+5):
                rnd_val = 0
                
            # Generate unique random numbers as cut indices:
                for j in range(5, no_cuts+5):
                    while rnd_val == 0 or rnd_val == auxiliary_arr[row, j]:
                        rnd = xoroshiro128p_uniform_float32(rng_states, row*auxiliary_arr.shape[1])\
                            *(auxiliary_arr[row, 3] - 2) + 2 # random*(max-min)+min
                        
                        rnd_val = int(rnd)+2            
                
                auxiliary_arr[row, i] = rnd_val
                
            # Sorting the crossover points:
            for i in range(5, no_cuts+5):
                min_index = i
                for j in range(i + 1, no_cuts+5):
                    # Select the smallest value
                    if auxiliary_arr[row, j] < auxiliary_arr[row, min_index]:
                        min_index = j

                auxiliary_arr[row, min_index], auxiliary_arr[row, i] = \
                auxiliary_arr[row, i], auxiliary_arr[row, min_index]
      
@cuda.jit
def crossOver(random_arr, auxiliary_arr, child_d_1, child_d_2, pop_d, parent_idx, crossover_prob):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, auxiliary_arr.shape[0], stride_x):
        for col in range(threadId_col, auxiliary_arr.shape[1] - 1, stride_y):
            if col > 1 and col < child_d_1.shape[1]-1:
                child_d_1[row, col] = pop_d[parent_idx[row, 0], col]
                child_d_2[row, col] = pop_d[parent_idx[row, 1], col]

                if random_arr[row, 0] <= crossover_prob: # Perform crossover with a probability of 0.6
                    no_cuts = auxiliary_arr[row, 4]

                    # Swap from first element to first cut point
                    if col < auxiliary_arr[row, 5]:
                        child_d_1[row, col], child_d_2[row, col] =\
                        child_d_2[row, col], child_d_1[row, col]

                    # For even number of cuts, swap from the last cut point to the end
                    if no_cuts%2 == 0:
                        if col > auxiliary_arr[row, no_cuts+4] and col < child_d_1.shape[1]-1:
                            child_d_1[row, col], child_d_2[row, col] =\
                            child_d_2[row, col], child_d_1[row, col]

                    # Swap in-between elements for 3+ cuts:
                    if no_cuts > 2:
                        for j in range(5, no_cuts+5):
                            cut_idx = auxiliary_arr[row, j]
                            if no_cuts%2 == 0:
                                if j%2==1 and col >= cut_idx and col < auxiliary_arr[row, j+1]:
                                    child_d_1[row, col], child_d_2[row, col] =\
                                    child_d_2[row, col], child_d_1[row, col]
                            
                            elif no_cuts%2 == 1:
                                if j%2==1 and col >= cut_idx and col < auxiliary_arr[row, j+1]:
                                    child_d_1[row, col], child_d_2[row, col] =\
                                    child_d_2[row, col], child_d_1[row, col]


# ------------------------------------Mutation part -----------------------------------------------
@cuda.jit
def inverseMutate(random_min_max, pop, random_no, mutation_prob):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)
    
    for row in range(threadId_row, pop.shape[0], stride_x):
        if random_no[row,0] <= mutation_prob:
            for col in range(threadId_col, pop.shape[1], stride_y):
                start  = random_min_max[row, 0]
                ending = random_min_max[row, 1]
                length = ending - start
                diff   = col - start
                if col >= start and col < start+ceil(length/2):
                    pop[row, col], pop[row, ending-diff] = pop[row, ending-diff], pop[row, col]

# -------------------------- Update population part -----------------------------------------------
@cuda.jit
def updatePop(count, parent_idx, child_d_1, child_d_2, pop_d):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y         = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):    
        for col in range(threadId_col, pop_d.shape[1], stride_y):

            if child_d_1[row, -1] <= pop_d[parent_idx[row, 0], -1] and \
            child_d_1[row, -1]    <= pop_d[parent_idx[row, 1], -1] and \
            child_d_1[row, -1]    <= child_d_2[row, -1]:

                pop_d[row, col] = child_d_1[row, col]

            elif child_d_2[row, -1] <= pop_d[parent_idx[row, 0], -1] and \
            child_d_2[row, -1]      <= pop_d[parent_idx[row, 1], -1] and \
            child_d_2[row, -1]      <= child_d_1[row, -1]:

                pop_d[row, col] = child_d_2[row, col]

            elif pop_d[parent_idx[row, 0], -1] <= pop_d[parent_idx[row, 1], -1] and \
            pop_d[parent_idx[row, 0], -1]      <= child_d_1[row, -1] and \
            pop_d[parent_idx[row, 0], -1]      <= child_d_2[row, -1]:

                pop_d[row, col] = pop_d[parent_idx[row, 0], col]

            elif pop_d[parent_idx[row, 1], -1] <= pop_d[parent_idx[row, 0], -1] and \
            pop_d[parent_idx[row, 1], -1]      <= child_d_1[row, -1] and \
            pop_d[parent_idx[row, 1], -1]      <= child_d_2[row, -1]:

                pop_d[row, col] = pop_d[parent_idx[row, 1], col]

            pop_d[row, 0]   = count
                
# ------------------------- Definition of CPU functions ----------------------------------------------   
def findMissingNodes(blocks, threads_per_block, data_d, pop, auxiliary_arr):
    reset_to_ones     [blocks, threads_per_block] (auxiliary_arr)
    prepareAuxiliary  [blocks, threads_per_block] (data_d, auxiliary_arr)
    findExistingNodes [blocks, threads_per_block] (data_d.shape[0], pop, auxiliary_arr)
    deriveMissingNodes[blocks, threads_per_block] (data_d, auxiliary_arr)

def generateCutPoints(blocks, threads_per_block, crossover_points, pop_d, popsize, auxiliary_arr):
    if crossover_points == 1:
        # assign cut points from the middle two quartiles
        auxiliary_arr[:, 5] = cp.random.randint((pop_d.shape[1]//4)*2, (pop_d.shape[1]//4)*3, size=popsize, dtype=cp.int32)
    else:
        rng_states = create_xoroshiro128p_states(popsize*pop_d.shape[1], seed=random.randint(2,2*10**5))
        add_cut_points[blocks, threads_per_block](auxiliary_arr, rng_states)    

def elitism(child_d_1, child_d_2, pop_d, popsize):

    # 5% from parents
    pop_d[0:int(0.05*popsize), :] = pop_d[pop_d[:, -1].argsort()][0:int(0.05*popsize),:]

    # Sort child 1 & child 2:
    child_d_1 = child_d_1[child_d_1[:,-1].argsort()]
    child_d_2 = child_d_2[child_d_2[:,-1].argsort()]

    # 45% from child 1, and 50% from child 2:  
    pop_d[int(0.05*popsize):int(0.5*popsize), :]  = child_d_1[0:int(0.45*popsize), :]    
    pop_d[int(0.5*popsize):popsize, :]            = child_d_2[0:int(0.5*popsize), :]

def showExecutionReport(count, start_time, best_sol):           
        end_time      = timer()
        total_time    = float('{0:.4f}'.format((end_time - start_time)))
        time_per_loop = float('{0:.4f}'.format((end_time - start_time)/(count-1)))

        val.validate(pop_d, 1)

        best_sol     = cp.subtract(best_sol, cp.ones_like(best_sol))
        best_sol[0]  = best_sol[0] + 1
        best_sol[-1] = best_sol[-1] + 1

        print('---------\nProblem:', sys.argv[1], ', Best known:', opt)
        print('Time elapsed:', total_time, 'secs', 'Time per loop:', time_per_loop, 'secs', end = '\n---------\n')
        print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
            %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n')
        print('Best solution:', best_sol, end = '\n---------\n')

        text_out = open('results/'+sys.argv[1]+str(datetime.now())+'.out', 'a')
        print('---------\nProblem:', sys.argv[1], ', Best known:', opt, file=text_out)
        print('Time elapsed:', total_time, 'secs', 'Time per loop:', time_per_loop, 'secs', end = '\n---------\n', file=text_out)
        print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
            %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n', file=text_out)
        print('Best solution:', best_sol, end = '\n---------\n', file=text_out)
        text_out.close()
# -----------------------------------------------------------------------------------------------------
def nCr(n,r):
    f = np.math.factorial
    return int(f(n) / (f(r) * f(n-r)))
# -----------------------------------------------------------------------------------------------------
def cleanUp(del_list):
    del del_list[:]
    
# ------------------------- Main Function ------------------------------------------------------------
try:
    vrp_capacity, data, opt = readInput()
    n                       = int(sys.argv[4])
    crossover_prob          = int(sys.argv[5])
    mutation_prob           = int(sys.argv[6])
    popsize                 = -(-(n*(data.shape[0] - 1))//1000)*1000
    crossover_points        = 1
    
    print('Taking population size {}*number of nodes'.format(n))
    
    try:
        generations = int(sys.argv[2])
    except:
        print('No generation limit provided, taking 2000 generations...')
        generations = 2000

    r_flag       = 9999 # A flag for removal/replacement
    data_d       = cuda.to_device(data)

    # Linear upper triangle of cost table (width=nC2))    
    linear_cost_table  = cp.zeros((nCr(data.shape[0], 2)), dtype=np.int32)
    pop_d              = cp.ones((popsize, int(1.5*data.shape[0])+2), dtype=np.int32)
    auxiliary_arr      = cp.zeros(shape=(popsize, pop_d.shape[1]), dtype=cp.int32)
    
    # GPU grid configurations:
    grid               = gpuGrid.GRID()
    blocks_x, blocks_y = grid.blockAlloc(data.shape[0], float(n))
    tpb_x, tpb_y       = grid.threads_x, grid.threads_y

    print(grid)
    blocks             = (blocks_x, blocks_y)
    threads_per_block  = (tpb_x, tpb_y)   

    val = val.VRP(sys.argv[1], data.shape[0])
    val.read()
    val.costTable()
    
    # --------------Calculate the cost table----------------------------------------------
    calculateLinearizedCost[blocks, threads_per_block](data_d, linear_cost_table)
    
    # --------------Initialize population----------------------------------------------
    initializePop[blocks, threads_per_block](data_d, pop_d)

    for individual in pop_d:
        cp.random.shuffle(individual[2:-1])

    find_duplicates[blocks, threads_per_block](pop_d, r_flag)
    shift_r_flag   [blocks, threads_per_block](r_flag, pop_d)

    cap_adjust    [blocks, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)
    cleanup_r_flag[blocks, threads_per_block](r_flag, pop_d)

    # --------------Calculate fitness----------------------------------------------
    # time_list = []
        
    # for i in range(1000):
    #     start_time = timer()
    #     pop_d[:, -1] = 0
    #     computeFitness[blocks, threads_per_block](linear_cost_table, pop_d, data_d.shape[0])
    #     end_time = timer()
    #     time_list.append(end_time - start_time)
    # print('Average time of new function: {} seconds +/- {}'.format(np.mean(time_list), np.std(time_list)))

    # for i in range(1000):
    #     start_time = timer()
    #     fitness_gpu_old[blocks, threads_per_block](linear_cost_table, pop_d, data_d.shape[0])
    #     end_time = timer()
    #     time_list.append(end_time - start_time)
    # print('Average time of old function: {} seconds +/- {}'.format(np.mean(time_list), np.std(time_list)))

    # exit()
    pop_d[:, -1] = 0
    computeFitness[blocks, threads_per_block](linear_cost_table, pop_d, data_d.shape[0])

    # ------------------------Evolve population for some generations------------------------
    parent_idx = cp.ones((popsize, 2), dtype=np.int32)
    child_d_1  = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
    child_d_2  = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
    cut_idx_d  = cp.ones(shape=(pop_d.shape[1])   , dtype=np.int32)
 
    del_list   = [data_d, linear_cost_table, pop_d, auxiliary_arr, parent_idx, child_d_1, child_d_2, cut_idx_d]

    minimum_cost = float('Inf')
    start_time   = timer()

    count         = 0
    best_sol      = 0
    total_time    = 0.0
    time_per_loop = 0.0

    while count <= generations:
        if minimum_cost <= opt:
            break

        random_arr_d = cp.arange(popsize, dtype=cp.int32).reshape((popsize,1))
        random_arr_d = cp.repeat(random_arr_d, 4, axis=1)
    
        for j in range(4):
            cp.random.shuffle(random_arr_d[:,j])
                
        # Select parents:
        selectParents   [blocks, threads_per_block](pop_d, random_arr_d, parent_idx)
        getParentLengths[blocks, threads_per_block](crossover_points, pop_d, auxiliary_arr, parent_idx)        
        generateCutPoints(blocks, threads_per_block, crossover_points, pop_d, popsize, auxiliary_arr)
        # print(auxiliary_arr[:10,:10])
        # cleanUp(del_list)
        # exit()
      
        random_arr = cp.random.randint(1, 100, (popsize, 1))
        crossOver[blocks, threads_per_block](random_arr, auxiliary_arr, child_d_1, child_d_2, pop_d, parent_idx, crossover_prob)

        # Performing mutation:
        random_min_max = cp.random.randint(2, pop_d.shape[1]-2, (popsize, 2))
        random_min_max.sort()
        random_no_arr  = cp.random.randint(1, 100, (popsize, 1))
        inverseMutate[blocks, threads_per_block](random_min_max, child_d_1, random_no_arr, mutation_prob)
        
        random_min_max = cp.random.randint(2, pop_d.shape[1]-2, (popsize, 2))
        random_min_max.sort()
        random_no_arr  = cp.random.randint(1, 100, (popsize, 1))
        inverseMutate[blocks, threads_per_block](random_min_max, child_d_2, random_no_arr, mutation_prob)       
           
        # time profiling old and new functions:
        # time_list = []
        
        # for i in range(1000):
        #     start_time = timer()
        #     find_duplicates[blocks, threads_per_block](child_d_1, r_flag)
        #     find_missing_nodes[blocks, threads_per_block](data_d, missing_d, child_d_1)
        #     add_missing_nodes[blocks, threads_per_block](missing_d, child_d_1, r_flag)
        #     end_time = timer()
        #     time_list.append(end_time - start_time)
        # print('Average time of new function: {} seconds +/- {}'.format(np.mean(time_list), np.std(time_list)))

        # for i in range(1000):
        #     start_time = timer()
        #     find_duplicates_old[blocks, threads_per_block](child_d_1, r_flag)
        #     find_missing_nodes_old[blocks, threads_per_block](data_d, missing_d, child_d_1)
        #     add_missing_nodes_old[blocks, threads_per_block](missing_d, child_d_1, r_flag)
        #     end_time = timer()
        #     time_list.append(end_time - start_time)
        # print('Average time of old function: {} seconds +/- {}'.format(np.mean(time_list), np.std(time_list)))

        # cleanUp(del_list)
        # exit()

        # Adjusting child_1 array:
        find_duplicates   [blocks, threads_per_block](child_d_1, r_flag)
        findMissingNodes  (blocks, threads_per_block, data_d, child_d_1, auxiliary_arr)
        addMissingNodes   [blocks, threads_per_block](r_flag, auxiliary_arr, child_d_1)
        shift_r_flag      [blocks, threads_per_block](r_flag, child_d_1)        
        cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_1)        
        cleanup_r_flag[blocks, threads_per_block](r_flag, child_d_1)

        # Adjusting child_2 array:
        find_duplicates   [blocks, threads_per_block](child_d_2, r_flag)
        findMissingNodes  (blocks, threads_per_block, data_d, child_d_2, auxiliary_arr)
        addMissingNodes   [blocks, threads_per_block](r_flag, auxiliary_arr, child_d_2)
        shift_r_flag      [blocks, threads_per_block](r_flag, child_d_2)
        cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_2)
        cleanup_r_flag[blocks, threads_per_block](r_flag, child_d_2)    

        # Performing the two-opt optimization and Calculating fitness for child_1 array:
        reset_to_ones[blocks, threads_per_block](auxiliary_arr)
        twoOpt      [blocks, threads_per_block](child_d_1, auxiliary_arr, linear_cost_table, data_d.shape[0])        
        
        child_d_1[:, -1] = 0
        computeFitness[blocks, threads_per_block](linear_cost_table, child_d_1, data_d.shape[0])
        
        # Performing the two-opt optimization and Calculating fitness for child_2 array:
        reset_to_ones[blocks, threads_per_block](auxiliary_arr)
        twoOpt      [blocks, threads_per_block](child_d_2, auxiliary_arr, linear_cost_table, data_d.shape[0])

        child_d_2[:, -1] = 0
        computeFitness[blocks, threads_per_block](linear_cost_table, child_d_2, data_d.shape[0])        

        # Creating the new population from parents and children:
        updatePop[blocks, threads_per_block](count, parent_idx, child_d_1, child_d_2, pop_d)
        elitism  (child_d_1, child_d_2, pop_d, popsize)

        # Picking the best and the worst solutions, population is already sorted at the elitism function:
        best_sol      = pop_d[0, :]
        minimum_cost  = best_sol[-1]        
        worst_cost    = pop_d[-1, :][-1]
        delta         = worst_cost-minimum_cost
        average       = cp.average(pop_d[:,-1])
        
        if count == 1:
            print('At first generation, Best: %d,'%minimum_cost, 'Worst: %d'%worst_cost, \
                'delta: %d'%delta, 'Avg: %.2f'%average)

        elif (count+1)%100 == 0:
            print('After %d generations, Best: %d,'%(count+1, minimum_cost), 'Worst: %d'%worst_cost, \
                'delta: %d'%delta, 'Avg: %.2f'%average)
        
        count += 1

    showExecutionReport(count, start_time, best_sol)   
    cleanUp(del_list)

except KeyboardInterrupt:
    showExecutionReport(count, start_time, best_sol)
    cleanUp(del_list)