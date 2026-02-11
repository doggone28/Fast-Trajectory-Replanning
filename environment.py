

#!this is all the gameboard shit, making a class for gameboarding and all that stuff


import numpy as np
import random

#*Our silly little robot world
#*We can model our world with a 2D array
#*Note that 0 represents an unblocked space, 1 represents a blocked space
#*According to project requirements, 
#* Part 0 - Setup your Environments [10 points]: You will perform all your experiments 
#* in the same 50 gridworlds of size 101 × 101
class GridWorld:


    #* 4 properties of this class: size (defauled to 101)
    #* the grid, start and end locations
    def __init__(self, size: int = 101):
        self.size = size
        self.grid = np.zeroes((size,size), dtype = int)
        self.start = None
        self.end = None


    

    