

#!special binary heap for keepign track of the a* open list

class BinaryHeap:from typing import Tuple, Any


class BinaryHeap:
    """
    Min-heap implementation for A* open list.
    Stores (priority, counter, state) tuples where:
    - priority: f-value (or modified f-value for tie-breaking)
    - counter: insertion order for stable tie-breaking
    - state: the state/cell
    """
    
    def __init__(self):
        self.heap = []
        self.entry_finder = {}  # Maps state -> [priority, counter, state, valid]
        self.counter = 0
    
    def push(self, state: Tuple[int, int], priority: float):
        """Add a new state or update the priority of an existing state"""
        if state in self.entry_finder:
            self.remove(state)
        
        entry = [priority, self.counter, state, True]
        self.entry_finder[state] = entry
        self.heap.append(entry)
        self.counter += 1
        self._sift_up(len(self.heap) - 1)
    
    def remove(self, state: Tuple[int, int]):
        """Mark an existing state as invalid"""
        if state in self.entry_finder:
            entry = self.entry_finder[state]
            entry[3] = False  # Mark as invalid
    
    def pop(self) -> Tuple[Tuple[int, int], float]:
        """Remove and return the state with lowest priority"""
        while self.heap:
            priority, _, state, valid = self._pop_min()
            if valid:
                del self.entry_finder[state]
                return state, priority
        raise KeyError('Pop from empty priority queue')
    
    def _pop_min(self):
        """Remove and return minimum element"""
        if not self.heap:
            raise KeyError('Pop from empty heap')
        
        min_entry = self.heap[0]
        last_entry = self.heap.pop()
        
        if self.heap:
            self.heap[0] = last_entry
            self._sift_down(0)
        
        return min_entry
    
    def _sift_up(self, pos: int):
        """Move element at pos up to maintain heap property"""
        entry = self.heap[pos]
        
        while pos > 0:
            parent_pos = (pos - 1) // 2
            parent = self.heap[parent_pos]
            
            if entry[0] < parent[0] or (entry[0] == parent[0] and entry[1] < parent[1]):
                self.heap[pos] = parent
                pos = parent_pos
            else:
                break
        
        self.heap[pos] = entry
    
    def _sift_down(self, pos: int):
        """Move element at pos down to maintain heap property"""
        entry = self.heap[pos]
        end_pos = len(self.heap)
        
        child_pos = 2 * pos + 1
        while child_pos < end_pos:
            # Choose smaller child
            right_pos = child_pos + 1
            if right_pos < end_pos:
                child = self.heap[child_pos]
                right = self.heap[right_pos]
                if right[0] < child[0] or (right[0] == child[0] and right[1] < child[1]):
                    child_pos = right_pos
            
            child = self.heap[child_pos]
            if entry[0] > child[0] or (entry[0] == child[0] and entry[1] > child[1]):
                self.heap[pos] = child
                pos = child_pos
                child_pos = 2 * pos + 1
            else:
                break
        
        self.heap[pos] = entry
    
    def is_empty(self) -> bool:
        """Check if heap is empty"""
        return len(self.entry_finder) == 0
    
    def __contains__(self, state: Tuple[int, int]) -> bool:
        """Check if state is in heap"""
        return state in self.entry_finder and self.entry_finder[state][3]
    
    def __len__(self) -> int:
        """Return number of valid entries"""
        return sum(1 for entry in self.entry_finder.values() if entry[3])


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])