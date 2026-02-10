# Uses all code to run the experiment.

from agent import AbstractAgent
from env import GridWorld
from baselines import *
from visualize import visualize_environment


def run_experiment(world:GridWorld, agent:AbstractAgent, num_trials:int):
    scores = []
    num_time_goal_reached = 0

    for trial in range(num_trials):
        world.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(world)
            _, reward, done = world.step(action)

            total_reward += reward

        scores.append(total_reward)

        agent_pos, goal_pos, _ = world.get_state()
        reached_goal = agent_pos == goal_pos
        if reached_goal:
            num_time_goal_reached += 1

        print(f"Trial {trial + 1}/{num_trials}, Total Reward: {total_reward}, Goal Reached: {reached_goal}")

    avg_score = sum(scores) / num_trials
    print(f"\nAverage Score over {num_trials} trials: {avg_score}")
    print(f"Number of times goal reached: {num_time_goal_reached} out of {num_trials}")




if __name__ == "__main__":
    # Example usage
    env = GridWorld(size=10, slip_prob=0.1, num_obstacles=5)

    print("\n============== Random Agent ==============")

    agent = RandomAgent() 
    run_experiment(env, agent, num_trials=10)

    state_vec = env.state_history  
    visualize_environment(10, state_vec, figure_title="Random Agent")

    print("\n============= Greedy Agent ==============")
    
    agent = GreedyAgent()
    run_experiment(env, agent, num_trials=10)

    state_vec = env.state_history  
    visualize_environment(10, state_vec, figure_title="Greedy Agent")



