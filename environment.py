

#!this is all the gameboard shit, making a class for gameboarding and all that stuff


import numpy as np
import random
from typing import Tuple


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

    def generateMaze(self, blockProb):

        visited = np.zeroes((self.size, self.size), dtype = bool)
        stack = []

        startXPos, startYPos = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
        visited[startXPos, startYPos] = True
        self.grid[startXPos, startYPos] = 0
        stack.append((startXPos, startYPos))


        while not np.all(visited):
            if stack:
                current = stack[-1]
                unvisitedNeighbors = self.unvisitedNeighbors(current, visited)
                if unvisitedNeighbors:
                    neighbor = random.choice(unvisitedNeighbors)
                    nx, ny = neighbor  

                    visited[nx, ny] = True

                    if random.random() < blockProb:
                        self.grid[nx, ny] = 1
                    else:
                        self.grid[nx, ny] = 0
                        stack.append(neighbor)

                else:
                    stack.pop()

            else:
                unvisited = np.argwhere(~visited)
                if len(unvisited) > 0:
                    nextCell = tuple(unvisited[random.randint(0, len(unvisited) - 1)])
                    visited[nextCell] = True
                    self.grid[nextCell] = 0
                    stack.append(nextCell)
                else:
                    break

    def unvisitedNeighbors(self, cell: Tuple[int,int], visited: np.ndarray) -> list:
        x, y = cell
        neighbors = []

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if (0 <= nx < self.size and 0 <= ny < self.size and not visited[nx, ny]):
                neighbors.append((nx, ny))

        return neighbors



    


    