import copy

import numpy as np

from agents.dice_game_agent import DiceGameAgent


class HybridAgent(DiceGameAgent):
    def __init__(self, game):
        super().__init__(game)
        self._penalty = game._penalty
        self.rerolls = 0

    def reset(self):
        self.rerolls = 0

    def calculate_dice_score(self, dice):
        temp_dice = copy.deepcopy(dice)
        uniques, counts = np.unique(temp_dice, return_counts=True)
        if np.any(counts > 1):
            temp_dice[np.isin(temp_dice, uniques[counts > 1])] = \
                [self.game._flip[x] for x in temp_dice[np.isin(temp_dice, uniques[counts > 1])]]
        temp_dice.sort()

        return np.sum(temp_dice)

    def play(self, state):
        states, game_over, reward, probabilities = self.game.get_next_states((), state)
        penalty = (self.rerolls + 1) * self._penalty
        dice_value = [self.calculate_dice_score(np.array(dice)) for dice in states]
        expected_score = sum(np.multiply(dice_value, probabilities)) - penalty
        current_score = self.calculate_dice_score(np.array(state)) - (self.rerolls * self._penalty)

        if state == (1, 1, 1) or state == (1, 1, 6) or expected_score < current_score:
            self.reset()
            return 0, 1, 2
        else:
            self.rerolls += 1
            return ()
