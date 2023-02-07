import numpy as np

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