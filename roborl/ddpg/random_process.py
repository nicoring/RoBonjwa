import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(self, shape, theta=0.15, sigma=0.2, dt=1e-2):
        self.theta = theta
        self.mu = 0
        self.sigma = sigma
        self.dt = dt
        self.last = 0
        self.shape = np.atleast_1d(shape)

    def sample(self):
        self.last = self.last + self.theta * (self.mu - self.last) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.randn(*self.shape)
        return self.last

    def reset(self, sigma=None):
        if sigma is not None:
            self.sigma = sigma
        self.last = 0
