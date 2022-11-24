from pprint import pprint
import numpy as np

from agents.dice_game_agent import DiceGameAgent
from dicegame.dice_game import DiceGame


class OneStepValueIterationAgent(DiceGameAgent):
    def __init__(self, game: DiceGame):
        """
        If your code does any pre-processing on the game, you can do it here.

        You can always access the game with self.game
        """
        super().__init__(game)
        self.gamma = 1.0
        self.penalty = self.game._penalty
        self.initialised = False
        self.state_mappings = {}

        _, self.values = self.perform_single_value_iteration()
        _, self.values = self.perform_single_value_iteration()
        self.state_to_action = {state: self.get_optimal_action(state) for state in self.game.states}

    def initialise_iteration(self):
        self.initialised = True

        for state in self.game.states:
            action_mappings = {}

            for action in self.game.actions[:-1]:
                states, game_over, reward, probabilities = self.game.get_next_states(action, state)
                state_mappings = {s: p for s, p in zip(states, probabilities)}
                action_mappings.update({action: state_mappings})

            action_mappings.update({self.game.actions[-1]: {state: 1}})
            self.state_mappings.update({state: action_mappings})

    def calculate_index_of_action(self, state, action):
        not_hold_all = (action != self.game.actions[-1])
        values = np.array([])

        for next_state in self.game.states:
            probability = self.state_mappings[state][action].get(next_state, 0)
            reward = self.game.final_scores[next_state] - self.game.final_scores[state] - (not_hold_all * self.penalty)
            v_function = self.values[next_state]
            values = np.append(values, np.multiply(probability, np.add(reward, np.multiply(self.gamma, v_function))))
        return values.sum()

    def perform_single_value_iteration(self):
        self.game: DiceGame

        if not self.initialised:
            self.initialise_iteration()
            return None, {state: 0 for state in self.game.states}

        new_values = {state: 0 for state in self.game.states}
        for current_state in self.game.states:
            action_indexes = np.array([self.calculate_index_of_action(current_state, act) for act in self.game.actions])
            new_values[current_state] = np.max(action_indexes)

        return None, new_values

    def get_optimal_action(self, state):
        action_indexes = {act: self.calculate_index_of_action(state, act) for act in self.game.actions}
        return max(action_indexes, key=lambda key: action_indexes[key])

    def play(self, state):
        """
        given a state, return the chosen action for this state
        at minimum you must support the basic rules: three six-sided fair dice

        if you want to support more rules, use the values inside self.game, e.g.
            the input state will be one of self.game.states
            you must return one of self.game.actions

        read the code in dicegame.py to learn more
        """
        return self.state_to_action[state]
