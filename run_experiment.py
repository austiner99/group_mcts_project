# Uses all code to run the experiment.

import matplotlib.pyplot as plt

from agent import AbstractAgent
from baselines import GreedyAgent, RandomAgent
from env import GridWorld
from mcts import MCTSAgent
from visualize import visualize_environment


def run_experiment(world: GridWorld, agent: AbstractAgent, num_trials: int):
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

    return scores, num_time_goal_reached


if __name__ == "__main__":
    # Example usage
    env = GridWorld(size=10, slip_prob=0.1, num_obstacles=5)
    NUM_TRIALS = 100

    print("\n============== Random Agent ==============")

    random_agent = RandomAgent()
    random_scores, random_success = run_experiment(env, random_agent, num_trials=NUM_TRIALS)

    random_state_vec = env.state_history
    visualize_environment(10, random_state_vec, figure_title="Random Agent")

    print("\n============= Greedy Agent ==============")

    greedy_agent = GreedyAgent()
    greedy_scores, greedy_success = run_experiment(env, greedy_agent, num_trials=NUM_TRIALS)

    greedy_state_vec = env.state_history
    visualize_environment(10, greedy_state_vec, figure_title="Greedy Agent")

    print("\n============= MCTS Agent - Random ==============")

    mcts_agent = MCTSAgent(iterations=500, exploration_param=1.4, rollout_depth=50, epsilon=1.0)
    mcts_scores, mcts_success = run_experiment(env, mcts_agent, num_trials=NUM_TRIALS)

    mcts_state_vec = env.state_history
    visualize_environment(10, mcts_state_vec, figure_title="MCTS Agent - Random")

    print("\n============== MCTS Agent - Epsilon-Greedy ==============")

    mcts_agent_eps = MCTSAgent(iterations=500, exploration_param=1.4, rollout_depth=50, epsilon=0.1)
    mcts_eps_scores, mcts_eps_success = run_experiment(env, mcts_agent, num_trials=NUM_TRIALS)

    mcts_eps_state_vec = env.state_history
    visualize_environment(10, mcts_eps_state_vec, figure_title="MCTS Agent - Epsilon-Greedy")

    # Box and whisker plot for score distribution
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [random_scores, greedy_scores, mcts_scores, mcts_eps_scores],
        tick_labels=["Random Agent", "Greedy Agent", "MCTS - Random", "MCTS - Epsilon-Greedy"],
    )
    plt.title('Comparison of Agent Scores\n(Press "q" to exit)')
    plt.ylabel("Total Reward")
    plt.grid()
    plt.show()

    # Bar plot for number of times goal reached
    plt.bar(
        ["Random Agent", "Greedy Agent", "MCTS - Random", "MCTS - Epsilon-Greedy"],
        [random_success, greedy_success, mcts_success, mcts_eps_success],
    )
    plt.title('Number of Times Goal Reached\n(Press "q" to exit)')
    plt.ylabel("Count")
    plt.grid(axis="y")
    plt.show()
