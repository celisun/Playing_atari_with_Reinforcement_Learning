# a simple replay memory for experience replay, 
# can save transitions and sample a random batch of transitions 
# out of memory

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory =[]
        self.position = 0

    def save(self, *args):               # save a trasition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position]=Transition(*args)
        self.position= (self.position+1) % self.capacity

    def sample(self, batch_size):       # select random batch of transitions from memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


