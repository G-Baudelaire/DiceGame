from agents.dice_game_agent import DiceGameAgent


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)