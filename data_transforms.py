import torch

class MinMaxScaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):

        x_std = (x - self.min) / (self.max - self.min)
        x_scaled = x_std * (self.max - self.min) + self.min

        return x_scaled

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self.scale = torch.zeros_like(self.mean)

    def __call__(self, x):

        x_std = (x - self.mean) / self.std

        return x_std
    
class PreserveZero(object):
    def __init__(self, max_abs):
        self.max_abs = torch.as_tensor(max_abs)

    def __call__(self, x):

        x_std = x / self.max_abs

        return x_std