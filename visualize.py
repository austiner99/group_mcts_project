# Displays current state of experiment while running. Show the grid board and images of the agent, obstacles, and goal.
# Can also generate figures for project presentation.

import matplotlib.pyplot as plt
import numpy as np
import random

def visualize_environment(size, state_vec):
    '''
    Function visualizes the grid environment based on the state vector, which includes the all agent's positions, goal positions, 
    and obstacles positions. It also shows the current location of all objects in clear colors, as well as the previous path of the
    agent (getting more faded the further back in time, up to 5 timesteps) and previous obstacle positions (faded gray) if a 
    previous state is provided. The states should update as we progress through the states, showing the movement of the agent and
    obstacles over time.
    
    :param size: size of the grid environment (e.g., 10 for a 10x10 grid)
    :param state_vec: List of states to visualize, where each state is a tuple (agent_pos, goal_pos, obstacles)
    '''
    plt.figure(figsize=(6, 6))
    if not state_vec:
        print("No states to visualize.")
        return
    for i in range(len(state_vec)):
        #initialize grid
        grid = np.zeros((size, size, 3), dtype=np.uint8) + 255  # White background
        #plot previous state with faded colors
        for j in range(5, 0, -1):
            if i-j >= 0:
                fade_value = 60+39*j  # 255/5 = 51, so each step back is 51 less in intensity
                agent_pos, goal_pos, obstacles = state_vec[i-j]
                # grid = np.zeros((size, size, 3), dtype=np.uint8) + 255
                for obs in obstacles:
                    grid[obs] = [255, fade_value, fade_value]  # Obstacles in red
                grid[agent_pos] = [fade_value, fade_value, 255]  # Agent in blue
        # plot current state with full colors
        agent_pos, goal_pos, obstacles = state_vec[i]
        # grid = np.zeros((size, size, 3), dtype=np.uint8) + 255
        grid[goal_pos] = [0, 255, 0]  # Goal in green
        for obs in obstacles:
            grid[obs] = [255, 0, 0]  # Obstacles in red
        grid[agent_pos] = [0, 0, 255]  # Agent in blue
        plt.imshow(grid)
        plt.title(f"State {i}")
        plt.axis('on')
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        plt.pause(1)
        # plt.close()
    
size = 10
state_vec = [((0, 0), (9, 9), [(2, 2), (3, 3)]), \
            ((1, 0), (9, 9), [(2, 1), (3, 4)]), \
            ((2, 0), (9, 9), [(1, 1), (4, 4)]), \
            ((3, 0), (9, 9), [(1, 2), (5, 4)]), \
            ((4, 0), (9, 9), [(1, 3), (4, 4)]), \
            ((4, 1), (9, 9), [(1, 4), (4, 3)]), \
            ((4, 2), (9, 9), [(1, 5), (3, 3)])]  # Example state vector with 5 states showing agent movement
visualize_environment(size, state_vec)