import numpy as np
from util import BinaryHeap
from environment import GridWorld
from agent_knowledge import AgentKnowledge
from typing import List, Tuple
from util import manhattan_distance

class ForwardAStar:
    """
    Repeated Forward A* from the pseudocode in Forward_A*_implementation.txt.
    Uses BinaryHeap from util.py for the OPEN list.
    The agent assumes unknown cells are unblocked (free-space assumption)
    and replans whenever it discovers a blocked cell on its path.
    """

    def __init__(self, gridworld: GridWorld):
        self.gridworld = gridworld
        self.size = gridworld.size
        self.knowledge = AgentKnowledge(self.size)

        # Persistent arrays reused across searches via counter mechanism (lines 15-17)
        self.g = np.full((self.size, self.size), float('inf'))
        self.search_arr = np.zeros((self.size, self.size), dtype=int)
        self.counter = 0

        # Per-search structures
        self.tree = {}
        self.open_list = None
        self.closed = None

        # Stats
        self.expanded_cells = 0
        self.num_searches = 0

    def heuristic(self, s: Tuple[int, int], goal: Tuple[int, int]) -> int:
        return manhattan_distance(s, goal)

    def _get_successors(self, s: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid successors based on agent's current knowledge (line 5: actions a in A(s))"""
        x, y = s
        successors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.knowledge.is_presumed_unblocked((nx, ny)):
                    successors.append((nx, ny))
        return successors

    def compute_path(self, s_goal: Tuple[int, int]) -> bool:
        """
        ComputePath procedure (pseudocode lines 1-13).
        Returns True if a path to s_goal was found, False otherwise.
        """
        while not self.open_list.is_empty():
            # Pop state with smallest f-value (line 3)
            s, f_value = self.open_list.pop()

            # Termination check (line 2): g(sgoal) <= min f in OPEN
            # f_value was the min before popping, so this is equivalent
            if self.g[s_goal] <= f_value:
                return True

            # Line 4: CLOSED := CLOSED ∪ {s}
            self.closed.add(s)
            self.expanded_cells += 1

            # Line 5: for all actions a in A(s)
            for succ in self._get_successors(s):
                # Lines 6-8: lazy initialization via counter
                if self.search_arr[succ] < self.counter:
                    self.g[succ] = float('inf')
                    self.search_arr[succ] = self.counter

                # Lines 9-13: relaxation
                new_g = self.g[s] + 1  # c(s, a) = 1 for unblocked moves
                if self.g[succ] > new_g:
                    self.g[succ] = new_g
                    self.tree[succ] = s
                    # BinaryHeap.push handles "if succ in OPEN remove it" (line 12)
                    f_val = new_g + self.heuristic(succ, s_goal)
                    self.open_list.push(succ, f_val)

        return False  # OPEN is empty, no path

    def _reconstruct_path(self, s_start: Tuple[int, int], s_goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Follow tree pointers from s_goal back to s_start (line 30)"""
        path = []
        current = s_goal
        while current != s_start:
            path.append(current)
            current = self.tree[current]
        path.append(s_start)
        path.reverse()
        return path

    def run(self) -> List[Tuple[int, int]]:
        """
        Main procedure (pseudocode lines 14-34).
        Returns the trajectory the agent actually walked, or None if unreachable.
        """
        s_start = self.gridworld.start
        s_goal = self.gridworld.goal

        # Line 15: counter := 0
        self.counter = 0
        # Lines 16-17: search(s) := 0 for all s  (already zeroed in __init__)
        self.expanded_cells = 0
        self.num_searches = 0

        # Observe from initial position
        self.knowledge.observe(s_start, self.gridworld)

        trajectory = [s_start]

        # Line 18: while sstart != sgoal
        while s_start != s_goal:
            # Line 19: counter := counter + 1
            self.counter += 1
            self.num_searches += 1

            # Lines 20-21
            self.g[s_start] = 0
            self.search_arr[s_start] = self.counter

            # Lines 22-23
            self.g[s_goal] = float('inf')
            self.search_arr[s_goal] = self.counter

            # Line 24: OPEN := CLOSED := empty
            self.open_list = BinaryHeap()
            self.closed = set()
            self.tree = {}

            # Line 25: insert sstart into OPEN with f-value g(sstart) + h(sstart)
            f_start = self.heuristic(s_start, s_goal)
            self.open_list.push(s_start, f_start)

            # Line 26: ComputePath()
            found = self.compute_path(s_goal)

            # Lines 27-29: if OPEN = empty => no path
            if not found:
                print("I cannot reach the target.")
                return None

            # Line 30: follow tree-pointers from sgoal to sstart,
            # move agent until it reaches sgoal or a path cell is blocked
            planned_path = self._reconstruct_path(s_start, s_goal)

            for i in range(1, len(planned_path)):
                next_pos = planned_path[i]

                # Check if the next cell is actually blocked in the real world
                if self.gridworld.isBlocked(next_pos):
                    # Line 32: update the increased action costs
                    self.knowledge.knowledge[next_pos] = 1
                    break

                # Move agent to next_pos
                s_start = next_pos
                trajectory.append(s_start)

                # Observe neighbours from new position
                self.knowledge.observe(s_start, self.gridworld)

                # If newly observed info reveals a block ahead, stop and replan
                path_blocked = False
                for j in range(i + 1, len(planned_path)):
                    if self.knowledge.is_known_blocked(planned_path[j]):
                        path_blocked = True
                        break
                if path_blocked:
                    break

            # Line 31: set sstart to current state of agent (already updated above)

        # Lines 33-34
        print("I reached the target.")
        return trajectory


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
    # Quick demo of Forward A* on a single gridworld
    gridworld = GridWorld(size=101)
    gridworld.generateMaze(0.3)
    gridworld.set_start_goal()
    print(f"Start: {gridworld.start}, Goal: {gridworld.goal}")

    solver = ForwardAStar(gridworld)
    trajectory = solver.run()

    if trajectory:
        print(f"Trajectory length: {len(trajectory)}")
        print(f"Total cells expanded: {solver.expanded_cells}")
        print(f"Number of A* searches: {solver.num_searches}")
