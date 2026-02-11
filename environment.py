

#!this is all the gameboard shit, making a class for gameboarding and all that stuff


import numpy as np
import random
from typing import Tuple
import pickle


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
        self.grid = np.zeros((size,size), dtype = int)
        self.start = None
        self.end = None

    def generateMaze(self, blockProb):

        visited = np.zeros((self.size, self.size), dtype = bool)
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
    


    def set_start_goal(self):
        """Randomly select unblocked start and goal positions"""
        unblocked = np.argwhere(self.grid == 0)
        if len(unblocked) < 2:
            raise ValueError("Not enough unblocked cells for start and goal")
        
        indices = random.sample(range(len(unblocked)), 2)
        self.start = tuple(unblocked[indices[0]])
        self.goal = tuple(unblocked[indices[1]])



    def isValid(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size
    
    def isBlocked(self, pos: Tuple[int, int]) -> bool:
        if not self.isValid(pos):
            return True
        return self.grid[pos] == 1
    
    def get_neighbors(self, pos: Tuple[int, int]) -> list:
        """Get valid neighbors (4-directional movement)"""
        x, y = pos
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # East, South, West, North
        
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if self.is_valid(neighbor):
                neighbors.append(neighbor)
        
        return neighbors



    def save(self, filename: str):
        """Save gridworld to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'grid': self.grid,
                'start': self.start,
                'goal': self.goal,
                'size': self.size
            }, f)
    
    @staticmethod
    def load(filename: str):
        """Load gridworld from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        gridworld = GridWorld(data['size'])
        gridworld.grid = data['grid']
        gridworld.start = data['start']
        gridworld.goal = data['goal']
        return gridworld
        


class AgentKnowledge:
    
    
    def __init__(self, size: int):
        self.size = size
        # -1 = unknown, 0 = unblocked, 1 = blocked
        self.knowledge = np.full((size, size), -1, dtype=int)
    
    def observe(self, pos: Tuple[int, int], gridworld: GridWorld):
    
        for neighbor in gridworld.get_neighbors(pos):
            if self.knowledge[neighbor] == -1:  # If not yet observed
                if gridworld.is_blocked(neighbor):
                    self.knowledge[neighbor] = 1
                else:
                    self.knowledge[neighbor] = 0
        
        # Also mark current position as unblocked (agent is standing on it)
        self.knowledge[pos] = 0
    
    def is_known_blocked(self, pos: Tuple[int, int]) -> bool:
        """Check if position is known to be blocked"""
        if not (0 <= pos[0] < self.size and 0 <= pos[1] < self.size):
            return True
        return self.knowledge[pos] == 1
    
    def is_presumed_unblocked(self, pos: Tuple[int, int]) -> bool:
        """Check if position is presumed unblocked (free-space assumption)"""
        if not (0 <= pos[0] < self.size and 0 <= pos[1] < self.size):
            return False
        return self.knowledge[pos] != 1  # Unknown or known unblocked





def generate_environments(num_envs: int = 50, size: int = 101, save_dir: str = "environments"):
    """Generate and save multiple gridworld environments"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_envs):
        print(f"Generating environment {i+1}/{num_envs}...")
        gridworld = GridWorld(size)
        gridworld.generateMaze(0.3)
        gridworld.set_start_goal()
        gridworld.save(f"{save_dir}/gridworld_{i:02d}.pkl")
    
    print(f"Generated {num_envs} environments in {save_dir}/")


if __name__ == "__main__":
    # Generate 50 environments
    generate_environments(num_envs=50, size=101)
