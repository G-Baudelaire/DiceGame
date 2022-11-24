from typing import Tuple, Dict

import numpy as np

from agents.dice_game_agent import DiceGameAgent
from dicegame.dice_game import DiceGame


class ValueIterationAgent(DiceGameAgent):
    def __init__(self, gamma=0.97, theta=0.0001):
        """
        If your code does any pre-processing on the game, you can do it here

        You can always access the game with self.game
        """
        super().__init__(DiceGame())
        self._theta, self._gamma = theta, gamma
        self._penalty = self.game._penalty
        self._v_of_states = np.array([i for i in range(len(self.game.states))])
        self._single_iteration()
        #
        # self._state_mappings = self._get_state_mapping()
        # self._v_of_states.update({self.game.states[-1]: 0})
        # self.value_iteration()
        # self._state_to_optimal_action = {state: self.get_optimal_action(state) for state in self.game.states}

    def _get_probabilities(self):
        actions_count, states_count = len(self.game.actions), len(self.game.states)
        empty_3d_array = np.zeros((actions_count, states_count, states_count))
        for i, action in enumerate(self.game.actions[:-1]):
            for j, initial_state in enumerate(self.game.states):
                states, _, _, probabilities = self.game.get_next_states(action, initial_state)
                for k, final_state in enumerate(self.game.states):
                    if final_state in states:
                        empty_3d_array[i][j][k] = probabilities[states.index(final_state)]

        for j, initial_state in enumerate(self.game.states):
            for k, final_state in enumerate(self.game.states):
                if initial_state == final_state:
                    empty_3d_array[actions_count - 1][j][k] = 1
        return empty_3d_array
    def _get_gamma_v_states(self):
        actions_count, states_count = len(self.game.actions), len(self.game.states)
        empty_3d_array = np.zeros((actions_count, states_count, states_count))
        return np.multiply(self._gamma, np.add(empty_3d_array, self._v_of_states))

    def _get_r_functions(self):
        actions_count, states_count = len(self.game.actions), len(self.game.states)
        empty_3d_array = np.zeros((actions_count, states_count, states_count))
        for i, action in enumerate(self.game.actions):
            for j, initial_state in enumerate(self.game.states):
                states, _, _, probabilities = self.game.get_next_states(action, initial_state)
                initial_score = self.game.final_scores[initial_state]
                for k, final_state in enumerate(self.game.states):
                    add_penalty = (action != self.game.actions[-1])
                    final_score = self.game.final_scores[final_state]
                    empty_3d_array[i][j][k] = final_score - (initial_score + (self._penalty * add_penalty))
        return empty_3d_array
    def _single_iteration(self):
        gamma_by_v = self._get_gamma_v_states()
        r = self._get_r_functions()
        matrix = np.add(r, gamma_by_v)
        probs = self._get_probabilities()
        outcome = np.matmul(matrix, probs)
        print(outcome)
    def _get_state_mapping(self) -> Dict[Tuple, Dict[Tuple, Dict[Tuple, float]]]:
        """
        :return: A dictionary mapping
        {initial_state: {action: {final_state: chance_to_get_final_state_from_initial_state}}}.
        """
        return {initial_state: self._get_action_mapping(initial_state) for initial_state in self.game.states}

    def _get_action_mapping(self, initial_state: Tuple[int, int, int]) -> Dict[Tuple, Dict[Tuple, float]]:
        """
        Map actions to dictionaries mapping final_states to their probabilities of occurring due to an action from an
        initial_state.
        :param initial_state: Tuple with the initial state of the dice.
        :return: A dictionary mapping {action: {final_state: chance_to_get_final_state_from_initial_state}}.
        """
        action_mappings = dict()

        for action in self.game.actions[:-1]:
            states, _, _, probabilities = self.game.get_next_states(action, initial_state)
            state_to_probability_mapping = {state: probability for state, probability in zip(states, probabilities)}
            action_mappings.update({action: state_to_probability_mapping})

        action_mappings.update({self.game.actions[-1]: {initial_state: 1}})
        return action_mappings

    # def _get_probabilities(self, initial_state: Tuple[int, int, int], action: Tuple[int, int, int]) -> Tuple[
    #     float, ...]:
    #     """
    #     Create tuple of floats with the probability of each final_state occurring due to an action from an
    #     initial_state.
    #     :param initial_state: Tuple with the initial state of the dice.
    #     :param action: Tuple with the dice to re-roll.
    #     :return: Tuple of probabilities.
    #     """
    #     return tuple(self._state_mappings[initial_state][action].get(i, 0) for i in self.game.states)

    def _get_rewards(self, initial_state: Tuple[int, int, int], action: Tuple[int, int, int]) -> Tuple[int, ...]:
        """
        Create tuple of integers with the reward for switching from an initial_state to a final_state while accounting
        for the penalty cost of a re-roll.
        :param initial_state: Tuple with the initial state of the dice.
        :param action: Tuple with the dice to re-roll.
        :return: Tuple of rewards values.
        """

        def reward_function(initial_state: Tuple[int, int, int], final_state) -> int:
            """
            Calculate reward for switching from initial_state to final_state.
            :param initial_state: Tuple with the initial state of the dice.
            :param final_state: Tuple with the final state of the dice.
            :return: Value of reward.
            """
            initial_dice_score = self.game.final_scores[initial_state]
            final_dice_score = self.game.final_scores[final_state]
            penalty = ((action != self.game.actions[-1]) * self._penalty)
            return final_dice_score - (initial_dice_score + penalty)

        return tuple(reward_function(initial_state, i) for i in self.game.states)

    def _get_v_of_states(self) -> Tuple[float, ...]:
        """
        :return: Tuple of V*(State_1),...,V*(State_N)
        """
        return tuple(self._v_of_states[i] for i in self.game.states)

    def calculate_index_of_action(self, initial_state: Tuple[int, int, int], action: Tuple[int, int, int]) -> float:
        """
        Calculate sum s' r (p(s', r|s, a)[r + gamma.V(s')])
        :param initial_state: Tuple with the initial state of the dice.
        :param action: Tuple with the dice to re-roll.
        :return: Value for an action given an initial state.
        """
        probabilities = self._get_probabilities(initial_state, action)
        rewards = self._get_rewards(initial_state, action)
        v_of_states = self._get_v_of_states()
        return np.multiply(probabilities, np.add(rewards, np.multiply(self._gamma, v_of_states))).sum()

    def _get_new_delta(self, new_v_of_states: Dict[Tuple[int, int, int], float]) -> float:
        """
        Calculate the greatest change between previous values and newly calculated values.
        :param new_v_of_states: Newly calculated V*(state) values.
        :return: Greatest difference in values.
        """
        old = [self._v_of_states[state] for state in self.game.states]
        new = [new_v_of_states[state] for state in self.game.states]
        return np.max(np.absolute(np.subtract(old, new)))

    def perform_single_value_iteration(self) -> Tuple[float, Dict[Tuple[int, int, int], float]]:
        """
        Perform one iteration of the values for V*(state) [v_of_states].
        :return: Tuple with the new values for delta and dictionary of states mapped to their V*(state) values.
        """
        new_v_of_states = dict()
        for state in self.game.states:
            max_index = np.max([self.calculate_index_of_action(state, action) for action in self.game.actions])
            new_v_of_states.update({state: max_index})
        return self._get_new_delta(new_v_of_states), new_v_of_states

    def value_iteration(self) -> None:
        """
        Loop that performs iterations for the V*(state) values while theta < greatest change of V*(state) value.
        """
        delta = float("inf")
        while self._theta < delta:
            delta, self._v_of_states = self.perform_single_value_iteration()

    def get_optimal_action(self, initial_state: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Calculate the optimal action to take from a given initial state.
        :param initial_state: Tuple with the initial state of the dice.
        :return: Tuple of action to take.
        """
        action_indexes = {action: self.calculate_index_of_action(initial_state, action) for action in self.game.actions}
        return max(action_indexes, key=lambda key: action_indexes[key])

    def play(self, state: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Lookup the optimal action to take based on the state of the dice.
        :param state: Current state of the dice.
        :return: Tuple of action to take.
        """
        return self._state_to_optimal_action[state]


ValueIterationAgent()
