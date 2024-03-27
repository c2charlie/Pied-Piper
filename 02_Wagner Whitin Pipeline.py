# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:03:59 2024

@author: Lili Zheng

The implemetation of Wanger-Whitin O(n^2) algorithm for the dynamic loc size problem.
The algorithm scales linearily with input in size.
The linear scaling is due to using the Planning Horizon Theorem in implementation.

"""

# =============================================================================
# MAIN PROGRAM
# =============================================================================

import itertools

class CumsumList:
    """Fast sums from index i to j, by storing cumulative sums."""
    
    def __init__(self, elements):
        self.cumsums = [0] * len(elements)
        partial_sum = 0
        for i, element in enumerate(elements):
            partial_sum += element
            self.cumsums[i] = partial_sum
            
    def sum_between(self, i, j):
        """Compute sum: elements[i] + elements[i+1] + ... + elements[j]"""
        
        if i > j: return 0
        top = self.cumsums[j] if j < len(self.cumsums) else self.cumsums[-1]
        low = self.cumsums[i - 1] if i >= 1 else 0
        return top - low
    
def wagner_whitin(demands, order_costs, holding_costs):
    """Compute the optimal cost and program for the dynamic lot size problem"""
    
    d, o, h = demands, order_costs, holding_costs
    # _validate_inputs(demands, order_costs, holding_costs)
    d_cumsum = CumsumList(d)
    assert d[-1] > 0, "Final demand should be positive"
    
    T = len(d) # Problem size
    F = {-1: 0} # Base case for resursion. F[t] = min cost from period 0 to t
    t_star_star = 0 # Highest period where minimum was achieved
    
    # Used to construct the solution
    cover_by = {} # cover_by[t] = j => cover period j to t by ordering at j
    
    # Main forward recursion loop. At time t, choose a period j to order from
    # to cover from period j to period t (inclusive on both ends)
    
    for t in range(len(demands)):
        
        # If there is no demand in period t
        if d[t] == 0:
            F[t] = F[t - 1]
            cover_by[t] = t
            continue
        
        # There is demand in period t
        assert d[t] > 0 
        
        s_t = 0 # Partial sum s_t(j) := h_j * d_{j+1, t} + s_t(j+1)
        min_args = [] # List of arguments for the min() function
        for j in reversed(range(t_star_star, t + 1)):
            s_t += h[j] * d_cumsum.sum_between(j + 1, t)
            min_args.append(o[j] + s_t + F[j - 1])
            
        # Index of minimum value in list 'min_args'
        argmin = min_args.index(min(min_args))
        
        
        # Update the t** variable - using the Planning Horizon Theorem
        t_star_star = max(t_star_star, t - argmin)
        
        # Update the variables for the cost and program
        F[t] = min_args[argmin]
        cover_by[t] = t - argmin
        
    # Construct the optimal solution. Order d[j:t+1] at time j to cover
    # demands in period from j to t (inclusive)
    t = T - 1
    solution = [0] * T
    while True:
        j = cover_by[t] # Get order time in period from j to t
        solution[j] = sum(d[j: t+1])
        t = j - 1 # Set t (period end) to (period start) minus one
        if j == 0:
            break # Break out if this period started at 0
        
    return {"cost": F[len(demands) - 1], "solution": solution}