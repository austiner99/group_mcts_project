# Displays current state of experiment while running. Show the grid board and images of the agent, obstacles, and goal.
# Can also generate figures for project presentation.

import matplotlib.pyplot as plt
import numpy as np
import random

# Example state for visualization
size = random.randint(10, 30)  # Example grid size

agent_pos = (random.randint(size//2, size-1), random.randint(size//2, size-1))
goal_pos = (random.randint(0, size//2), random.randint(0, size//2))
if agent_pos == goal_pos:
    goal_pos = ((goal_pos[0] + 1) % size, goal_pos[1])  # Ensure agent and goal are not the same
obstacles = set()
while len(obstacles) < 10:
    pos = (random.randint(0, size-1), random.randint(0, size-1))
    if pos != agent_pos and pos != goal_pos and pos not in obstacles:
        obstacles.add(pos)

prev_agent_pos = (agent_pos[0]-1, agent_pos[1]) if agent_pos[0] > 0 else (agent_pos[0], agent_pos[1]-1)
prev_obstacles = set()
for obs in obstacles:
    new_obs = (obs[0]-1, obs[1]) if obs[0] > 0 else (obs[0], obs[1]-1)
    prev_obstacles.add(new_obs)
prev_state = (prev_agent_pos, goal_pos, tuple(prev_obstacles))  # Example previous state; can be set to visualize movement
state = (agent_pos, goal_pos, tuple(obstacles))  # Example state: (agent_pos, goal_pos, obstacles)

def visualize_environment(size, state, prev_state=None):
    agent_pos, goal_pos, obstacles = state
    grid = np.zeros((size, size, 3), dtype=np.uint8) + 255  # White background

    #if previous state is given, show the movement with a faded blue trail, and faded gray for previous obstacle positions
    if prev_state:
        prev_agent_pos, _, prev_obstacles = prev_state
        grid[prev_agent_pos[1], prev_agent_pos[0]] = [173, 216, 230]  # Light blue for previous position
        for obs in prev_obstacles:
            grid[obs[1], obs[0]] = [211, 211, 211]  # Light gray for previous obstacles

    # Draw obstacles in black
    for obs in obstacles:
        grid[obs[1], obs[0]] = [0, 0, 0]

    # Draw goal in green
    grid[goal_pos[1], goal_pos[0]] = [0, 255, 0]

    # Draw agent in blue
    grid[agent_pos[1], agent_pos[0]] = [0, 0, 255]

    plt.imshow(grid)
    ax = plt.gca()
    ax.add_patch(plt.Rectangle((-0.5, -0.5), size, size, fill=False, edgecolor='black', linewidth=2))
    plt.axis('off')
    plt.show()
    

visualize_environment(size, state, prev_state)