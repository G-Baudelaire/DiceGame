import time
from pprint import pprint

import numpy as np

from agents import perfectionist, always_hold, hybrid_agent, manual_agent, one_step_value_iteration_agent, \
    value_iteration_agent
from dicegame.dice_game import DiceGame


def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()

    if (verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if (verbose): print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        if (verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if (verbose and not game_over): print(f"Dice: \t\t{state}")

    if (verbose): print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score


if __name__ == "__main__":
    # random seed makes the results deterministic
    # change the number to see different results
    #  or delete the line to make it change each time it is run
    np.random.seed(1)
    game = DiceGame()

    theta_to_score = dict()
    for i in range(5):
        np.random.seed(1)
        theta = 1 / (10 ** i)
        print(f"Agent of theta {theta}")

        t_start = time.time()
        agent = value_iteration_agent.ValueIterationAgent(game, theta=theta)
        print(f"Time to initialise {time.time() - t_start}")

        t_start = time.time()
        score = np.mean([play_game_with_agent(agent, game) for games in range(100000)])
        print(f"Time to run {time.time() - t_start}")

        theta_to_score.update({theta: score})
        print(f"Average score of {score} over 100k games.\n")
    print(max(theta_to_score, key=lambda key: theta_to_score[key]))
