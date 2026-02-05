# Implementation of the grid world, including obstacles and goal and stochastic movement and dynamics pulled from config.py

import random

global MOVEMENT_REWARD, OBSTACLE_PENALTY, GOAL_REWARD, RANDOM_OBSTACLE_MOVE_PROB #variable sthat can be modified from config.py
MOVEMENT_REWARD = -1 # Penalty for each movement to encourage shorter paths
OBSTACLE_PENALTY = -10 # Penalty for hitting an obstacle
GOAL_REWARD = 20 # Reward for reaching the goal
RANDOM_OBSTACLE_MOVE_PROB = 0.7  # Probability that an obstacle will move at each time step

class GridWorld:
    def __init__(self, size=10, slip_prob=0.1, num_obstacles=0):
        self.size = size
        self.slip_prob = slip_prob
        self.num_obstacles = num_obstacles
        self.action_space = ['up', 'down', 'left', 'right']
        self.reset()
        
    def reset(self):
        self.agent_pos = [0, 0]  # START AT TOP-LEFT CORNER - COULD BE MODIFIED TO RANDOM START
        self.goal_pos = [self.size - 1, self.size - 1]  # Goal at bottom-right corner - COULD BE MODIFIED TO RANDOM GOAL
        self.obstacles = self.generate_obstacles()
        return self.agent_pos
        
    def generate_obstacles(self):
        obs = set()
        while len(obs) < self.num_obstacles:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if pos != tuple(self.agent_pos) and pos != tuple(self.goal_pos) and pos not in obs:
                obs.add(pos)
        return obs
    
    def get_state(self):
        return (tuple(self.agent_pos), tuple(self.goal_pos), tuple(map(tuple, self.obstacles)))
    
    def step(self, action):
        agent_position, goal_position, obstacles = self.get_state()
        self.agent_pos = list(agent_position)
        self.obstacles = [list(obs) for obs in obstacles]
        
        if random.random() < self.slip_prob:
            action = random.choice(self.action_space)
            
        self.move_agent(action)
        self.move_obstacles()
        
        reward = MOVEMENT_REWARD  
        done = False
        if self.agent_pos in self.obstacles:
            reward = OBSTACLE_PENALTY  
            done = True
        elif self.agent_pos == self.goal_pos:
            reward = GOAL_REWARD  
            done = True
            
        return self.get_state(), reward, done
    
    def move_agent(self, action):
        if action == 'down' and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 'up' and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1
        elif action == 'left' and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 'right' and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1
            
    def move_obstacles(self):
        for obs in self.obstacles:
            if random.random() < RANDOM_OBSTACLE_MOVE_PROB:
                direction = random.choice(self.action_space)
                if direction == 'up' and obs[1] > 0 and [obs[0], obs[1] - 1] != self.agent_pos and [obs[0], obs[1] - 1] != self.goal_pos:
                    obs[1] -= 1
                elif direction == 'down' and obs[1] < self.size - 1 and [obs[0], obs[1] + 1] != self.agent_pos and [obs[0], obs[1] + 1] != self.goal_pos:
                    obs[1] += 1
                elif direction == 'left' and obs[0] > 0 and [obs[0] - 1, obs[1]] != self.agent_pos and [obs[0] - 1, obs[1]] != self.goal_pos:
                    obs[0] -= 1
                elif direction == 'right' and obs[0] < self.size - 1 and [obs[0] + 1, obs[1]] != self.agent_pos and [obs[0] + 1, obs[1]] != self.goal_pos:
                    obs[0] += 1