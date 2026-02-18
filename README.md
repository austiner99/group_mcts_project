# Group Monte Carlo Tree Search (MCTS) Project
## Algorithms for Decision Making Class
### Description of Problem and Reasoning for MCTS Implementation
This repository is for the code for this project and for coordination amongst group members. 

The idea behind this project is to create a grid world with an agent navigating from one corner of the world to the other to reach a goal. In the agent's way are moving obstacles. If the agent ever occupies the same space as an obstacle, it loses (though we have found that obstacles and agents can occasionally "swap" squares with no penalty. This was left in as a fun bug, seeing as it is like the agent leaping over its would-be captors). You can think of this problem as similar to the "frogger" game. All initial locations and movement of obstacles is stochastic (random), along with a small probability for the agent to move contrary to its "chosen" direction. 

The MCTS-controlled agent will navigate the world using a MCTS simulation to select a movement (up, down, left or right) at each timestep using an upper confidence bound (UCB) strategy. This method was selected as it seemed the best way to deal with a large amount of stochasticity, and offered many parameters to adjust in order to optimize performance.

Baseline agents will also navigate the world using both a random movement algorithm and a greedy movement algorithm. The random algorithm selects a location at random, whereas the greedy algorithm always attempts to move in the shortest direction to the goal. These agent's performance will be compared to the MCTS agent in order to evaluate the usefulness of a MCTS algorithm in this problem.

Experimental results suggest that MCTS outperforms baseline methods under moderate-to-high stochasticity. Furthermore, the MCTS algorithm is able to find the goal much more on any given run than either the greedy or random algorithm (assuming sufficient obstacles are present). This can be seen in the figures generated upon running the algorithms. Note that the animation of each run is taken from each algorithm's best (highest scoring) run. Further interpretation of results can be seen below.

This project uses information from chapters 3, 5, 6 and 10 of [Algorithms for Decision Making](http://algorithmsbook.com/decisionmaking/) by Mykel J. Kochenderfer, Tim A. Wheeler, and Kyle H. Wray.

### Table of Contents
1. [env.py](env.py)
2. [mcts_random.py](mcts_random.py)
3. [mcts_uct.py](mcts_uct.py)
4. [baselines.py](baselines.py)
5. [run_experiment.py](run_experiment.py)
6. [visualize.py](visualize.py)
7. [config.py](config.py)
8. [agent.py](agent.py)

### Description of files

1. env.py - implementation of the grid world, including obstacles and goal and stochastic movement and parameters pulled from config.py

2. mcts_random.py - Monte Carlo code and sim for the random search. This chooses actions and updates statistics to be plugged into the enviornment base on "current" states. The random MCTS chooses random paths to search the space.

3. mcts_uct.py - Monte Carlo code and sim for the the UCT search. This chooses actions and updates statistics to be plugged into the enviornment base on "current" states. The UCT MCTS uses the upper confidence bound to pick which paths to search. 

4. baselines.py - code for random and greedy policies for comparison with mcts algorithms

5. run_experiment.py - uses all code to run the experiment 

6. visualize.py - displays animation of best runs of each agent type and generates plots of scores and success

7. config.py - sets of various parameters that can be used in each run

8. agent.py - creates an abstract agent class to better generalize running different agents in code.

### Instructions

1. Install dependencies in requirements.txt.

2. Run [run_experiment.py](run_experiment.py).

3. A visualization of a trial with each agent will pop up. The blue square is the agent, red are obstacles, and green is the goal. You can exit by pressing 'q'. Note: The MCTS agents will take a few minutes to run.

4. After the visualizations have been completed, a box and wisker plot will show the distribution of scores for each agent. 

5. After the previous plot is closed, a bar plot will show the number of times each agent successfully reached the goal without hitting an obstacle. 