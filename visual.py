import matplotlib
"""
Visualization utilities for Fast Trajectory Replanning
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Tuple
from environment import GridWorld, AgentKnowledge


def visualize_gridworld(gridworld: GridWorld, trajectory: List[Tuple[int, int]] = None,
                       knowledge: AgentKnowledge = None, title: str = "Gridworld"):
    """
    Visualize the gridworld with optional trajectory and agent knowledge.
    
    Args:
        gridworld: The gridworld to visualize
        trajectory: Optional list of positions showing agent's path
        knowledge: Optional agent knowledge to show what agent knows
        title: Title for the plot
    """
    fig, axes = plt.subplots(1, 3 if knowledge else 2, figsize=(15 if knowledge else 10, 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Plot 1: Actual gridworld
    ax1 = axes[0]
    ax1.imshow(gridworld.grid, cmap='binary', origin='upper')
    ax1.set_title("Actual Gridworld")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    
    # Mark start and goal
    if gridworld.start:
        ax1.plot(gridworld.start[1], gridworld.start[0], 'go', markersize=15, label='Start')
    if gridworld.goal:
        ax1.plot(gridworld.goal[1], gridworld.goal[0], 'r*', markersize=20, label='Goal')
    
    # Plot trajectory
    if trajectory and len(trajectory) > 0:
        traj_array = np.array(trajectory)
        ax1.plot(traj_array[:, 1], traj_array[:, 0], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        ax1.plot(traj_array[:, 1], traj_array[:, 0], 'bo', markersize=5, alpha=0.5)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Agent knowledge (if provided)
    if knowledge:
        ax2 = axes[1]
        # Create knowledge visualization (-1=grey, 0=white, 1=black)
        knowledge_vis = np.ones_like(knowledge.knowledge, dtype=float) * 0.5
        knowledge_vis[knowledge.knowledge == 0] = 1.0  # Unblocked = white
        knowledge_vis[knowledge.knowledge == 1] = 0.0  # Blocked = black
        
        ax2.imshow(knowledge_vis, cmap='gray', origin='upper', vmin=0, vmax=1)
        ax2.set_title("Agent's Knowledge")
        ax2.set_xlabel("Column")
        ax2.set_ylabel("Row")
        
        if gridworld.start:
            ax2.plot(gridworld.start[1], gridworld.start[0], 'go', markersize=15, label='Start')
        if gridworld.goal:
            ax2.plot(gridworld.goal[1], gridworld.goal[0], 'r*', markersize=20, label='Goal')
        
        if trajectory and len(trajectory) > 0:
            traj_array = np.array(trajectory)
            ax2.plot(traj_array[:, 1], traj_array[:, 0], 'b-', linewidth=2, alpha=0.7)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Presumed unblocked (what agent assumes)
        ax3 = axes[2]
        presumed = np.ones_like(knowledge.knowledge, dtype=float)
        presumed[knowledge.knowledge == 1] = 0.0  # Only known blocked shown as black
        
        ax3.imshow(presumed, cmap='binary', origin='upper')
        ax3.set_title("Presumed Unblocked (Agent's View)")
        ax3.set_xlabel("Column")
        ax3.set_ylabel("Row")
        
        if gridworld.start:
            ax3.plot(gridworld.start[1], gridworld.start[0], 'go', markersize=15, label='Start')
        if gridworld.goal:
            ax3.plot(gridworld.goal[1], gridworld.goal[0], 'r*', markersize=20, label='Goal')
        
        if trajectory and len(trajectory) > 0:
            traj_array = np.array(trajectory)
            ax3.plot(traj_array[:, 1], traj_array[:, 0], 'b-', linewidth=2, alpha=0.7)
        
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        # Just plot trajectory on second axis if no knowledge provided
        ax2 = axes[1]
        ax2.imshow(gridworld.grid, cmap='binary', origin='upper', alpha=0.3)
        ax2.set_title("Trajectory")
        ax2.set_xlabel("Column")
        ax2.set_ylabel("Row")
        
        if gridworld.start:
            ax2.plot(gridworld.start[1], gridworld.start[0], 'go', markersize=15, label='Start')
        if gridworld.goal:
            ax2.plot(gridworld.goal[1], gridworld.goal[0], 'r*', markersize=20, label='Goal')
        
        if trajectory and len(trajectory) > 0:
            traj_array = np.array(trajectory)
            ax2.plot(traj_array[:, 1], traj_array[:, 0], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            ax2.plot(traj_array[:, 1], traj_array[:, 0], 'bo', markersize=5)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_animation(gridworld: GridWorld, trajectory: List[Tuple[int, int]], 
                     knowledge_history: List[AgentKnowledge], filename: str = None):
    """
    Create an animation showing the agent's movement through the gridworld.
    
    Args:
        gridworld: The gridworld environment
        trajectory: List of positions the agent visited
        knowledge_history: List of agent's knowledge at each step
        filename: Optional filename to save animation (e.g., 'animation.gif')
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Left plot: actual gridworld with trajectory so far
        ax1.imshow(gridworld.grid, cmap='binary', origin='upper')
        ax1.set_title(f"Actual Gridworld - Step {frame}")
        ax1.plot(gridworld.start[1], gridworld.start[0], 'go', markersize=15, label='Start')
        ax1.plot(gridworld.goal[1], gridworld.goal[0], 'r*', markersize=20, label='Goal')
        
        if frame > 0:
            traj = np.array(trajectory[:frame+1])
            ax1.plot(traj[:, 1], traj[:, 0], 'b-', linewidth=0.5, alpha=0.7)
            # Current position
            ax1.plot(trajectory[frame][1], trajectory[frame][0], 'bo', markersize=3)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: agent's knowledge
        if frame < len(knowledge_history):
            knowledge = knowledge_history[frame]
            presumed = np.ones_like(knowledge.knowledge, dtype=float)
            presumed[knowledge.knowledge == 1] = 0.0
            
            ax2.imshow(presumed, cmap='binary', origin='upper')
            ax2.set_title(f"Agent's Knowledge - Step {frame}")
            ax2.plot(gridworld.start[1], gridworld.start[0], 'go', markersize=15, label='Start')
            ax2.plot(gridworld.goal[1], gridworld.goal[0], 'r*', markersize=20, label='Goal')
            
            if frame > 0:
                traj = np.array(trajectory[:frame+1])
                ax2.plot(traj[:, 1], traj[:, 0], 'b-', linewidth=0.5, alpha=0.7)
                ax2.plot(trajectory[frame][1], trajectory[frame][0], 'bo', markersize=3)
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=500, repeat=True)
    
    if filename:
        anim.save(filename, writer='pillow', fps=2)
    
    plt.tight_layout()
    return anim


def plot_comparison(results: dict, title: str = "Algorithm Comparison"):
    """
    Plot comparison of different algorithms.
    
    Args:
        results: Dictionary with algorithm names as keys and statistics as values
        title: Title for the plot
    """
    algorithms = list(results.keys())
    metrics = ['total_expansions', 'num_searches', 'trajectory_length']
    metric_names = ['Total Expansions', 'Number of Searches', 'Trajectory Length']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[alg][metric] for alg in algorithms]
        axes[i].bar(algorithms, values)
        axes[i].set_title(name)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test visualization with sample data
    from environment import GridWorld
    
    # Create simple gridworld
    gridworld = GridWorld(size=101)
    gridworld.generateMaze(0.3)
    gridworld.set_start_goal()
    
    print(f"Start: {gridworld.start}, Goal: {gridworld.goal}")
    
    # Create sample trajectory
    trajectory = [gridworld.start]
    current = gridworld.start
    # Simple trajectory toward goal
    while current != gridworld.goal:
        dx = np.sign(gridworld.goal[0] - current[0])
        dy = np.sign(gridworld.goal[1] - current[1])
        if dx != 0:
            current = (current[0] + dx, current[1])
        elif dy != 0:
            current = (current[0], current[1] + dy)
        trajectory.append(current)
        if len(trajectory) > 100:  # Safety
            break
    
    # Visualize
    fig = visualize_gridworld(gridworld, trajectory=trajectory, title="Test Gridworld")
    plt.show()