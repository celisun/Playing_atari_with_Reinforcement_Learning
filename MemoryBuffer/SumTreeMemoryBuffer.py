import numpy as np
from SumTree import SumTree
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))



class SumTreeMemoryBuffer(object):
    """Prioritized experience replay,
         implemented by sum tree"""

    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def save(self, data):
    # Store transition to memory
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, data)

    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, self.tree.data[0].size))#b_memory = np.empty((n, self.tree.data[0].size))
        ISWeights = np.empty((n, 1))
        pri_seg = self.tree.total_p / n

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        temp = self.tree.tree[-self.tree.capacity:]
        temp_nonzero = temp[np.nonzero(temp)]   # calculate non-zero min p
        min_prob = np.min(temp_nonzero) / self.tree.total_p

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i+1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.sample(v)
            prob = p / self.tree.total_p
            #print ('min P')
            #print (min_prob)
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        #return b_memory
        return b_idx, b_memory, ISWeights      # [N,] [N,18] [N,1]


    def batch_update(self, tree_idx, abs_errors):

        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper) #clip td
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.length

    @property
    def data(self):
        return self.tree.data
    @property
    def gettree(self):
        return self.tree.tree
