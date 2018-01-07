# A Sum Tree data structure
# implemented for prioritized replay memory database: SumTreeMemoryBuffer
import numpy as np


class SumTree(object):
    
    data_pointer = 0
    length = 0
    
    def __init__ (self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity -1)        # parent nodes: capacity-1, leaves: capacity
        self.data = np.zeros(capacity, dtype=object)      # data is transition
         
            
    def add(self, p, data): 
    # add node
    # args: priority p and transition data       
        tree_idx = self.data_pointer + self.capacity -1
        self.data[self.data_pointer] = data      # update data frame 
        self.update(tree_idx, p)
        
        self.data_pointer += 1
        self.length += self.length
        if self.data_pointer >= self.capacity:     # replace when memory full
            self.data_pointer = 0
        
        
    def update(self, tree_idx, p):
    # update a node with new key (td-error)
    # used by add operation
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # percolate 
        while tree_idx != 0:
            tree_idx = (tree_idx - 1)//2
            self.tree[tree_idx] += change 
        
        
    def sample(self, v):
    # prioritized sample from memoory
        parent_idx = 0
        while True: 
            cl_idx = 2 * parent_idx + 1
            cr_idx = parent_idx + 1
            
            if cl_idx >= len(self.tree):  # reach bottom, end search 
                leaf_idx = parent_idx
                break
            else:     # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx 
                else: 
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx 
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
                 
    @property
    def total_p(self):
    # get sum
        return self.tree[0]  # root 
        