import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt
import random
import numpy as np

from matplotlib.animation import FuncAnimation
import seaborn as sns

# Set random seeds

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Define data loading step

batch_size = 256

def create_data(samples,start,end,coff1,coff2,coff3,coff4):
    data = []
    for sample in range(samples):
        theta = random.uniform(start,end)
        x = np.cos(2*np.pi*theta)
        y = np.sin(2*np.pi*theta)
        #code = [0,0,0,0,0]
        #index = random.randint(0,4)
        #code[index] = 1
        point = [coff1*np.cos(-2*theta)+coff2*np.cos(-theta)+coff3*np.cos(theta)+coff4*np.cos(2*theta), coff1*np.sin(-2*theta)+coff2*np.sin(-theta)+coff3*np.sin(theta)+coff4*np.sin(2*theta)]
        data.append([point[0],point[1]])
        #data.append(code)
    return data
    
kwargs = {'num_workers': 1, 'pin_memory': True}
#train_loader = create_data(10000)
'''
train_loader = torch.utils.data.DataLoader(
    MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
'''
# Defining the device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defining the model

d = 20

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, d * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            #nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, 2)).view(-1, 2, d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

model = VAE().to(device)
model = model.double()

# Setting the optimiser

learning_rate = 1e-3

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

# Reconstruction + β * KL divergence losses summed over all elements and batch
def loss_function(x_hat, x, mu, logvar, β=0.0001):
    
    BCE = nn.functional.mse_loss(
        x_hat, x.view(-1, 2)
    )
    
    #BCE = nn.functional.binary_cross_entropy(
    #    x_hat, x.view(-1, 5), reduction='sum'
    #)
    
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE + β * KLD
    #return BCE

# Training and testing the VAE
#epochs = 300
#for epoch in range(0, epochs + 1):
epoch = -1
n = 0
train_loader = create_data(10000,n*2*np.pi/3,n*2*np.pi/3+2*np.pi/3,1,0,-1,0)
gen_loader = torch.empty((0))
def animate(t):
    global epoch, n, train_loader, gen_loader
    # Training
    if epoch > 0:  # test untrained net first
        if ((epoch % 100) == 0):
            n +=1
            train_loader = create_data(10000,n*2*np.pi/3,n*2*np.pi/3+2*np.pi/3,1,0,-1,0)
            gen_loader = torch.empty((0))
            for i in range(100):
                z = torch.randn((100, d)).to(device)
                z = torch.tensor(z,dtype=torch.double)
                gen_loader = torch.concat([gen_loader,model.decoder(z)],axis=0)
                
        train_loss = 0
        random.shuffle(train_loader)
        #train_loader = create_data(100)
        for batch in range(int(len(train_loader)/batch_size)):
            model.train()
        
            x = train_loader[batch*batch_size:(batch+1)*batch_size]
            
            if n > 0:
                #random.shuffle(gen_loader)
                x_gen = gen_loader[batch*batch_size:(batch+1)*batch_size]
                prec_gen = 0.5
                x = torch.tensor(x,dtype=torch.double)
                x = torch.concat([x[:int(prec_gen*batch_size)],x_gen[:int((1-prec_gen)*batch_size)]],axis=0)
                
            
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

    xp = np.linspace(-3,3,300)
    yp = np.linspace(-3,3,300)
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
    #plt.imshow(a, cmap='jet', vmin = 0, vmax = 0.25)
    #plt.show()
    sns.heatmap(a,
                ax = ax,
                cbar = True,
                cbar_ax = cbar_ax,
                cmap='jet',
                vmin = 0,
                vmax =0.25)
    epoch+=1
    

grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))
anim = FuncAnimation(fig = fig, func = animate, frames = 300, interval = 50, blit = False)
#plt.show()


anim.save('vae_energy.gif',
          writer = 'Pillow', fps = 10)

