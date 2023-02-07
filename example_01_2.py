# Example 01
# A simple environment to build understanding of creating a mental model of the world and then using it to predict future movement.
# Enivornment: Peter the particle moves around the circfurmence of a circle in the orgin of xy plane, Peter only senses its angle 
# relative to the x axis in radians. Peter moves at a constant velocity of 2pi rad/sec and senses at 10hz.
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from motion_models import Particle
from sensory_motor_inference import WorldModel

class PredictionNet(nn.Module):
    
    def __init__(self,network):
        super().__init__()
        
        self.input = nn.Sequential(
            nn.Linear(1, 1)
        )
        
        self.network = network
        
    def forward(self,x):
        p =  self.input(torch.ones(1))
        x  = torch.tensor([x])
        xp  = torch.concat([x,p])
        input = torch.tensor(xp,dtype=torch.double)
        x_hat, mu, logvar = self.network(input)
        energy = (x-x_hat[0][0])**2 + (p-x_hat[0][1])**2
        return energy, xp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = WorldModel().to(device)
model = model.double()
model.load_state_dict(torch.load('saved_models/ex1'))

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = PredictionNet(model).to(device)

for index, param in enumerate(network.parameters()):
    if index > 0:
        param.requires_grad = False


# Setting the optimiser
learning_rate = 1e-1
optimizer = torch.optim.Adam(
    network.parameters(),
    lr=learning_rate,
)

for i in range(100):
    quality, pstate = network.forward(2*np.pi-1*2*np.pi/10)
            
    # ===================backward====================
    optimizer.zero_grad()
    quality.backward()
    optimizer.step()
    print("quality: ",quality)
    print("pstate: ",pstate)

xp = np.linspace(-1,7,300)
yp = np.linspace(-1,7,300)
grid = []
for x in xp:
    for y in yp:
        grid.append([x,y])
input = torch.tensor(grid,dtype=torch.double)
x_hat, mu, logvar = model(input)
a = np.zeros((300, 300))
for xidx in range(300):
    for yidx in range(300):
        index = xidx*300+yidx
        loss = (grid[index][0]-x_hat[index][0])**2 + (grid[index][1]-x_hat[index][1])**2 
        a[xidx][yidx] = loss

peter = Particle()
markers = peter.generate_samples(10)
print(markers)
for marker in markers:
    x_index = marker[0]
    for yidx in range(300): 
        i_index = int(300*(x_index - -1)/(7 - -1))
        a[i_index][yidx] = 0

#plot prediction marker
x_index = int(300*(pstate[0] - -1)/(7 - -1))
y_index = int(300*(pstate[1] - -1)/(7 - -1))
dis = 15
deltas = range(-dis,dis,1)
for dx in deltas:
    a[x_index+dx][y_index] = 0
for dy in deltas:
    a[x_index][y_index+dy] = 0

fig, ax = plt.subplots()
sns.heatmap(a,
            ax = ax,
            cbar = True,
            cmap='jet',
            vmin = 0,
            vmax = 0.25)

plt.show()