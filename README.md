# Group Monte Carlo Tree Search (MCTS) Project
## Algorithms for Decision Making Class
### Description
This repository is for all the code for this group project and for coordination amongst group members. 

The idea behind this project is to create a grid world with an agent navigating the world to a (potentially moving) goal. In the agent's way are obstacles who also may move. All potential movement of objects and the goal(s) will be stochastic (random). 

The agent will navigate the world using a MCTS simulation to select a movement (up, down, left or right) at each timestep using an upper confidence bound (UCB) strategy (this can be amended based on group input). 

The algorithm can be evaluated against random and greedy algorithms (not totally sure how this would work yet). The project uses informaion from chapters 3, 5, 6 and 10 of the textbook for this course.

### Table of Contents
1. [env.py](env.py)
2. [mcts_random.py](mcts_random.py)
3. [mcts_uct.py](mcts_uct.py)
4. [baselines.py](baselines.py)
5. [run_experiment.py](run_experiment.py)
6. [visualize.py](visualize.py)
7. [config.py](config.py)

### Description of files

1. env.py - implementation of the grid world, including obstacles and goal and stochastic movement and dynamics pulled from config.py

2. mcts_random.py - Monte Carlo code and sim for the random search. This chooses actions and updates statistics to be plugged into the enviornment base on "current" states. The random MCTS chooses random paths to search the space.

3. mcts_uct.py - Monte Carlo code and sim for the the UCT search. This chooses actions and updates statistics to be plugged into the enviornment base on "current" states. The UCT MCTS uses the upper confidence bound to pick which paths to search. 

4. baselines.py - code for random and greedy policies for comparison with mcts algorithm

5. run_experiment.py - uses all code to run the experiment 

6. visualize.py - displays current state of experiment while running. Can also generate figures for project presentation

7. config.py - functions to dictate how enviornment/agent reacts to movement/how things move around. Kind of the "miscellaneous" file

### Instructions

1. Install dependencies in requirements.txt.

2. Run [run_experiment.py](run_experiment.py).

3. A visualization of a trial with each agent will pop up. The blue square is the agent, red are obstacles, and green is the goal. You can exit by pressing 'q'. Note: The MCTS agents will take a few minutes to run.

4. After the visualizations have been completed, a box and wisker plot will show the distribution of scores for each agent. 

5. After the previous plot is closed, a bar plot will show the number of times each agent successfully reached the goal without hitting an obstacle. 