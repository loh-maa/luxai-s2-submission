
import numpy as np
import sys
from typing import Optional

from a09_3a import board_routines
from a09_3a import utils
from a09_3a.board_routines import BoardManager
from a09_3a.genome import Genome
from a09_3a.logger import log
from a09_3a.plant_manager import PlantManager
from a09_3a.robot_manager import RobotManager
from a09_3a.robot import Robot
from base.agent import Agent

from lux.config import EnvConfig
from lux.kit import obs_to_game_state, GameState


class AgentOne(Agent):
    def __init__(self, player: str, env_cfg: EnvConfig, genome: Optional[Genome] = None, episode_id: int = 0):
        super().__init__(player, env_cfg)

        self.episode_id = episode_id

        # Default genome
        self.genome = genome or Genome.loads()

        # todo implement power sharing

        # This will print out in agent logs
        print(f'{player}: a09_3a genome loaded: {self.genome.id}', file=sys.stderr)
        log.info(f'{player}: a09_3a genome loaded: {self.genome.id}')

        self.plant_man = PlantManager.new_instance(player, env_cfg, self.genome)
        self.robot_man = None

        # Initialize static constants
        board_routines.DistanceMatrix.initialize(env_cfg.map_size)

        # Keep the latest game_state (extended)
        self.game_state = None

    def get_name(self):
        return f'a09_3a-{self.genome.id}'

    def get_stats(self):
        stats = {
            'agent': self.get_name(),
            'player': self.player,
            # Episode ID is the same as seed, with that negative indicate reversed players.
            'episode_id': self.episode_id
        }
        if hasattr(self, 'game_state') and self.game_state is not None:
            stats['step'] = self.game_state.real_env_steps
            stats['our_lichen'] = np.sum(board_routines.get_lichen_map(self.game_state, self.player))
            stats['opp_lichen'] = np.sum(board_routines.get_lichen_map(self.game_state, self.opp_player))

        stats.update(Robot.stats)

        stats['meff'] = self.robot_man.mission_efficiency_stats

        return stats

    def is_development(self):
        return self.episode_id != 0

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        game_state: GameState = obs_to_game_state(step, self.env_cfg, obs)
        game_state = utils.game_state_extended(game_state, self.player, self.opp_player, episode_id=self.episode_id)
        self.game_state = game_state

        actions = self.plant_man.early_setup(game_state, remainingOverageTime)
        return actions

    def act(self, step0: int, obs, remainingOverageTime: int = 60):
        log.debug(f'remainingOverageTime: {remainingOverageTime}')

        game_state = obs_to_game_state(step0, self.env_cfg, obs)
        game_state = utils.game_state_extended(game_state, self.player, self.opp_player, episode_id=self.episode_id)
        self.game_state = game_state

        units = game_state.units[self.player]
        actions = dict()

        # Overwrite the step with real step (the same as in logging and visualizer)
        step = game_state.real_env_steps
        log.debug(f'@{step} --------------------------------------------------------------------------------------')

        # If that's the last step, log the stats, maybe also lichen
        if step == 999:
            log.info(f'@{step} {self.player} internal stats: {self.get_stats()}')

        if self.robot_man is None:
            self.robot_man = RobotManager(game_state, self.genome)
            BoardManager.new_instance(game_state)

        BoardManager.get_instance().new_step(game_state)

        self.plant_man.update_plants(game_state)

        if step % 20 == 0:
            self.plant_man.update_arable_stats(game_state)

        # Factory actions
        # ---------------
        actions_mod = self.plant_man.plan_watering(game_state, self.robot_man.home_factory_robots)
        actions.update(actions_mod)

        # Plan missions
        # -------------
        robot_actions = self.robot_man.plan_missions(game_state, remainingOverageTime)
        actions.update(robot_actions)

        # Action filters:
        actions = self.robot_man.apply_action_filter(actions)

        # Try to resolve collisions, at least 5 iterations
        self.robot_man.update_next_locations(game_state, units, actions)
        resolved = False
        n_attempts = int(np.clip(round((remainingOverageTime / 60)**2 * 5), 1, 5))
        for i in range(n_attempts):

            # todo is it actually benecifial!!?
            actions_mod_aut = self.robot_man.avoid_guarded_res_tiles(game_state)
            if actions_mod_aut:
                actions.update(actions_mod_aut)
                # Update only locations of units with modified actions
                mod_units = {uid: units[uid] for uid in actions_mod_aut.keys()}
                self.robot_man.update_next_locations(game_state, mod_units, actions)

            actions_mod_opp = self.robot_man.collision_avoidance_with_opponent_units(game_state)
            if actions_mod_opp:
                actions.update(actions_mod_opp)
                # Update only locations of units with modified actions
                mod_units = {uid: units[uid] for uid in actions_mod_opp.keys()}
                self.robot_man.update_next_locations(game_state, mod_units, actions)

            # Try to ram only on the first iteration
            actions_mod_ram = {}
            if i == 0 and self.genome.flags['ramming_enabled']:
                actions_mod_ram = self.robot_man.ram_opponent_units(game_state)
                if actions_mod_ram:
                    actions.update(actions_mod_ram)
                    # Update only locations of units with modified actions
                    mod_units = {uid: units[uid] for uid in actions_mod_ram.keys()}
                    self.robot_man.update_next_locations(game_state, mod_units, actions)

            actions_mod_our = self.robot_man.collision_avoidance_with_our_units(game_state, actions)
            if actions_mod_our:
                actions.update(actions_mod_our)
                # Update only locations of units with modified actions
                mod_units = {uid: units[uid] for uid in actions_mod_our.keys()}
                self.robot_man.update_next_locations(game_state, mod_units, actions)

            if not (actions_mod_aut or actions_mod_our or actions_mod_opp or actions_mod_ram):
                # No more avoidance moves found
                resolved = True
                break

        if not resolved:
            log.debug(f'@{step} collisions unresolved')

        # Final action filter:
        actions = self.robot_man.apply_action_filter(actions)

        n_lights_total = sum([len(self.robot_man.home_factory_robots[1].get(fid, [])) for fid in game_state.factories[self.player].keys()])
        log.debug(f'total light robots: {n_lights_total}')

        # We can produce new units only if no robots are expected at factory center locations, always overwrite watering
        for fid, factory in game_state.factories[self.player].items():
            if factory.loc not in Robot.next_locs.values():
                plant = PlantManager.get_instance().get_plant(fid)

                n_heavy = len(self.robot_man.home_factory_robots[10].get(fid, []))
                n_light = len(self.robot_man.home_factory_robots[1].get(fid, []))

                wait = (self.robot_man.get_l2h_ratio(fid)**2 + plant.wait_for_metal +
                        len(self.robot_man.home_factory_robots[1].get(fid, [])))

                wait = wait > 0 or factory.cargo.metal + factory.cargo.ore / 5 >= 100 or (step < 100 and n_heavy == 1)

                # todo if the factory has no source of ore, it can build lights straight away
                ore_div = plant.get_ore_diversity()
                block = ore_div > 0.1 and (self.robot_man.get_l2h_ratio(fid) > 3 or (n_heavy < 2 and step < 120)) and step < 810

                allow = self.robot_man.get_l2h_ratio(fid) < 1.5 and step > 200

                if factory.can_build_heavy(game_state):
                    actions[fid] = factory.build_heavy()

                # We prefer heavy robots, unless there's a significant disproportion..
                # check to see if we should wait for more ore
                elif (block or wait or n_lights_total > 100) and not n_light == 0 and not allow:
                    # Wait until the factory processes enough ore
                    continue

                elif factory.can_build_light(game_state):
                    actions[fid] = factory.build_light()

        if self.is_development():
            self.robot_man.account_for_actions(game_state)

        return actions

