import numpy as np

class ReplayBuffer(object):
    def __init__(self, algorithm, max_size=1000000):
        """
        :param max_size (int): total amount of tuples to store
        """
        self.algorithm = algorithm
        
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.cntr = 0

    def add(self, data):
        """
        Add experience tuples to buffer
        
        :param data (tuple): experience replay tuple
        """
        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
        self.cntr += 1
            
    def save(self, name='ReplayBuffer'):
        '''Save replay buffer for live trading'''
        self.algorithm.ObjectStore.Save(name, str(self.storage))
        self.algorithm.Debug("{} - Saving Replay Buffer!: {}".format(self.algorithm.Time, len(self.storage)))
    
    def load(self, name='ReplayBuff'):
        '''Load replay buffer for live trading'''
        self.storage = eval(self.algorithm.ObjectStore.ReadBytes("key"))
        self.algorithm.Debug("{} - Loading Replay Buffer!: {}".format(self.algorithm.Time, len(self.storage)))

    def sample(self, batch_size):
        """
        Samples a random amount of experiences from buffer of batch size
        
        :param batch_size (int): size of sample
        """
        
        index = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in index: 
            s, s_, a, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            next_states.append(np.array(s_, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(next_states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)