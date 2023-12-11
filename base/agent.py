import numpy as np

from lux.config import EnvConfig
from lux.kit import obs_to_game_state


class Agent:

    def __init__(self, player: str, env_cfg: EnvConfig):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    def get_name(self):
        return 'unknown'

    def get_stats(self):
        return {
            'agent': self.get_name()
        }

    def save_data(self, score):
        pass

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        # optionally convert observations to python objects with utility functions
        # game_state = obs_to_game_state(step, self.env_cfg, obs)
        return actions

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        # game_state = obs_to_game_state(step, self.env_cfg, obs)
        return actions

