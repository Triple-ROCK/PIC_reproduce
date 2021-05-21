import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.act_shape = args.act_shape
        self.n_agents = self.args.n_agents
        self.obs_shape = self.args.obs_shape
        # memory management
        self.ptr, self.size, self.max_size = 0, 0, self.args.buffer_size
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.max_size, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.max_size, self.n_agents, self.act_shape]),
                        'r': np.empty([self.max_size, 1]),
                        'o_next': np.empty([self.max_size, self.n_agents, self.obs_shape]),
                        'done': np.empty([self.max_size, 1])
                        }

    def store(self, obs, act, rew, next_obs, done):
        """
        store one transition into buffer
        """
        self.buffers['o'][self.ptr] = obs
        self.buffers['u'][self.ptr] = act
        self.buffers['r'][self.ptr] = rew
        self.buffers['o_next'][self.ptr] = next_obs
        self.buffers['done'][self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        temp_buffer = {}
        batch_idx = np.random.randint(0, self.size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][batch_idx]
        return temp_buffer
