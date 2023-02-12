import random
import math
from sympy import re

# class SD(): # Plain () for python 3.x
#     def __init__(self):
#         self.sum, self.sum2, self.n = (0,0,0)
#         self.var = 0
#     def sd(self, x):
#         self.sum  += x
#         self.sum2 += x*x
#         self.n    += 1.0
#         sum, sum2, n = self.sum, self.sum2, self.n

#         self.var = math.sqrt(sum2/n - sum*sum/n/n)

#         return self.var

# class Welford:
#     def __init__(self):
#         self.mean = 0
#         self.M2 = 0
#         self.k = 0
#         self.var = 0

#     def update(self, x):
#         self.k += 1
#         delta = x - self.mean
#         self.mean += delta / self.k
#         self.M2 += delta * (x - self.mean)
#         if self.k > 1:
#             self.var = self.M2 / (self.k - 1)

class Welford:
    def __init__(self):
        self.mean_L = 0
        self.mean_l = 0
        self.M2 = 0
        self.k = 0
        self.var_l = 0
        self.z = 0
        self.alpha = 1

    def update(self, loss):
        self.k += 1

        if self.k > 1:
            self.l = loss / self.mean_L
        else:
            self.l = 0
        
        delta_L = loss - self.mean_L
        delta_l = self.l - self.mean_l
        
        self.mean_L += delta_L / self.k
        self.mean_l += delta_l / self.k 

        self.M2 += delta_l * (self.l - self.mean_l)
        if self.k > 1:
            self.var_l = self.M2 / (self.k - 1)
    
    def get_alpha(self):

        if self.k > 1:
            c = self.var_l/self.mean_l
            self.z += c
            self.alpha = (1/self.z)*c
            
        return self.alpha



# class TemperatureSensor:
#     def __init__(self):
#         self.w = SD()
#         self.t = []

#     def read(self):
#         # simulate reading a temperature
#         temperature = 20 + 10*random.random()
#         self.w.sd(temperature)
#         self.t.append(temperature)
#         return temperature

# sensor = TemperatureSensor()

# for i in range(100):
#     temperature = sensor.read()
#     #print(f"Temperature: {temperature}, Mean: {sensor.w.mean}, Variance: {sensor.w.variance()}")
#     print(f"Temperature: {temperature}, Variance: {sensor.w.var}")

# print('hey')

import torch    
from torch import nn

class Welford_MSE(nn.Module):

    def __init__(self):
        super(Welford_MSE, self).__init__()
        
        self.w = Welford()
        self.l_fn = torch.nn.MSELoss()

    def loss(self, y_hat, y):

        l = self.l_fn(y_hat, y)

        self.w.update(l.detach().item())
       
        return self.w.get_alpha() * l

input = torch.tensor([1,2,3],dtype=torch.float64,requires_grad=True)
output = torch.tensor([4,5,6],dtype=torch.float64,requires_grad=True)

l_fn = Welford_MSE()


a=l_fn.loss(output,input)
a=l_fn.loss(input,output)
a=l_fn.loss(output,input)

print('a')