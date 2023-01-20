import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from matplotlib.animation import FuncAnimation

# generate random noise for the heatmap
#rnd_data = np.random.normal(0, 1, (500, 100, 100))

def get_point(theta):
    point = [0.5*np.cos(-2*theta)+0*np.cos(-theta)+0.5*np.cos(theta)+0*np.cos(2*theta), 0.5*np.sin(-2*theta)+0*np.sin(-theta)+0.5*np.sin(theta)+0*np.sin(2*theta)]
    return  point

def get_distance(x,y):
    min_dist = None
    tp = np.linspace(0,2*np.pi,100)
    for theta in tp:
        point = get_point(theta)
        px = point[0]
        py = point[1]
        dist = (px-x)**2+(py-y)**2
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return np.sqrt(min_dist)
    
    
y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

#z = np.cos(2*np.pi*0.5*x)*np.cos(2*np.pi*0.5*y)


z  = np.ones((100,100))

for ix, xp in enumerate(np.linspace(-3, 3, 100)):
    for iy, yp  in enumerate(np.linspace(-3, 3, 100)):
        z[iy,ix] = get_distance(xp,yp)

fig = plt.figure()
axis = plt.axes(xlim =(-3, 3),
                    ylim =(-3, 3))

sns.heatmap(z,
            ax = axis,
            cbar = True,
            cmap='jet',
            vmin = 0,
            vmax = 1)

plt.show()




