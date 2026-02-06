# Uses all code to run the experiment.

from agent import AbstractAgent
from env import GridWorld
from baselines import *


def run_experiment(world:GridWorld, agent:AbstractAgent, num_trials:int):
    scores = []

    for trial in range(num_trials):
        world.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(world)
            _, reward, done = world.step(action)

            total_reward += reward

        scores.append(total_reward)
        print(f"Trial {trial + 1}/{num_trials}, Total Reward: {total_reward}")

    avg_score = sum(scores) / num_trials
    print(f"Average Score over {num_trials} trials: {avg_score}")




if __name__ == "__main__":
    # Example usage
    env = GridWorld(size=10, slip_prob=0.1, num_obstacles=5)

    print("============== Random Agent ==============")

    agent = RandomAgent() 
    run_experiment(env, agent, num_trials=10)

    print("============== Greedy Agent ==============")
    
    agent = GreedyAgent()
    run_experiment(env, agent, num_trials=10)


