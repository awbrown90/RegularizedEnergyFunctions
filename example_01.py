# Example 01
# A simple environment to build understanding of creating a mental model of the world and then using it to predict future movement.
# Enivornment: Peter the particle moves around the circfurmence of a circle in the orgin of xy plane, Peter only senses its angle 
# relative to the x axis in radians. Peter moves at a constant velocity of 2pi rad/sec and senses at 10hz.
import torch
from torch import nn
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

class Particle:
    
    def __init__(self):
        self.pos = 0
        self.sense_hz = 10
        self.step_size = 2*np.pi/self.sense_hz
        
    def move(self):
        self.pos += 1
        if (self.pos >= self.sense_hz):
            self.pos = 0
        
    def sense(self):
        return self.pos * self.step_size
    
    def generate_samples(self,samples):
        data = []
        for i in range(samples):
            point = []
            point.append(self.sense())
            self.move()
            point.append(self.sense())
            data.append(point)
        return data
    

class WorldModel(nn.Module):
    def __init__(self):
        self.d = 10
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, self.d * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.d, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, 2)).view(-1, 2, self.d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

peter = Particle()
train_loader = peter.generate_samples(10000)
#print(train_loader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = WorldModel().to(device)
model = model.double()

# Setting the optimiser
learning_rate = 2e-2
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

# Reconstruction + β * KL divergence losses summed over all elements and batch
def loss_function(x_hat, x, mu, logvar, β=0):#0.000001):
    BCE = nn.functional.mse_loss(
        x_hat, x.view(-1, 2)
    )
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return BCE + β * KLD

# Learning the world model
load_model = True
train_model = False
if load_model:
    print("loading model")
    model.load_state_dict(torch.load('saved_models/ex1'))
if train_model:
    batch_size = 256
    model.train()
    for epoch in range(300):
        # Training

        train_loss = 0
        random.shuffle(train_loader)

        for batch in range(int(len(train_loader)/batch_size)):
            x = train_loader[batch*batch_size:(batch+1)*batch_size]
            x = train_loader
            x = torch.tensor(x,dtype=torch.double)
            x = x.to(device)

            # ===================forward=====================
            x_hat, mu, logvar = model(x)
            loss = loss_function(x_hat, x, mu, logvar)
            train_loss += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
        print(f'====> Epoch: {epoch} Average loss: {train_loss / int(len(train_loader)/batch_size):.4f}')
        
model.eval()

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

markers = peter.generate_samples(10)
print(markers)
for marker in markers:
    x_index = marker[0]
    for yidx in range(300): 
        i_index = int(300*(x_index - -1)/(7 - -1))
        a[i_index][yidx] = 0

fig, ax = plt.subplots()
sns.heatmap(a,
            ax = ax,
            cbar = True,
            cmap='jet',
            vmin = 0,
            vmax = 0.25)

plt.show()

save_model = False
if save_model:
    print("saving model")
    torch.save(model.state_dict(), 'saved_models/ex1')


