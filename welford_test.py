import random
import math

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

#     def update(self, x):
#         self.k += 1
#         delta = x - self.mean
#         self.mean += delta / self.k
#         self.M2 += delta * (x - self.mean)

#     def variance(self):
#         if self.k > 1:
#             return self.M2 / (self.k - 1)
#         else:
#             return 0

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



class Welford:
    def __init__(self) -> None:
        self.u_L = 0
        self.u_l = 0
        self.l = 0
        self.M = 0
        self.t = 0
        self.s_l = 0
        self.c = 0
        self.z = 0
    
    def update(self, loss):
        self.t += 1
        if self.t == 1:
            self.l = 0
        else:
            self.l = loss / self.u_L
        self.u_L = (1 - 1 / self.t) * self.u_L + (1/self.t) * loss
        u_l = (1 - 1 / self.t) + (1/self.t) * self.l
        self.M = (1-1/self.t) * self.M + (1/self.t) * (self.l-self.u_l) * (self.l-u_l)
        self.u_l = u_l
        self.s_l = math.sqrt(self.M)
    
    def get_alpha(self):
        if self.t == 1:
            self.c = 0
        else:
            self.c = self.u_l/self.s_l
        
        self.z += self.c

        if self.t == 1:
            return 1
        else:
            return (1/self.z) * self.c



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


l_fn.loss(output,input)

print('a')