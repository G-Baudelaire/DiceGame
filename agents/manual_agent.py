import copy

import numpy as np

from agents.dice_game_agent import DiceGameAgent


class ManualAgent(DiceGameAgent):

    def __init__(self, game):
        super().__init__(game)
        self._penalty = game._penalty
        self.rerolls = 0

    def reset(self):
        self.rerolls = 0

    def calculate_dice_score(self, dice):
        return self.game.final_scores[dice]

    def calculate_expected_score(self, action, state, total_penalty):
        states, _, _, probabilities = self.game.get_next_states(action, state)
        scores = np.array([self.calculate_dice_score(dice) for dice in states])
        return np.sum(np.multiply(scores, probabilities)) - total_penalty

    def play(self, state):
        total_penalty = (self.rerolls + 1) * self._penalty
        current_score = self.calculate_dice_score(state) - (self.rerolls * self._penalty)
        expected_scores = np.array(
            [self.calculate_expected_score(action, state, total_penalty) for action in self.game.actions[:-1]])
        expected_scores = np.append(expected_scores, current_score)
        optimal_action = self.game.actions[np.argmax(expected_scores)]

        if optimal_action == (0, 1, 2):
            self.reset()
            return optimal_action
        else:
            self.rerolls += 1
            return optimal_action
