from cachetools import cached, LRUCache
from cachetools.keys import hashkey
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from a09_3a import board_routines
from a09_3a import utils
from a09_3a.logger import log
from a09_3a.path_finding import NhoodMatrix
from a09_3a.plant import Plant

from lux.kit import GameState


class ScoreLayers(IntEnum):
    # Score factors/layers
    RUBBLE = 0
    # How much lichen can we grow right away without removing any rubble
    ARABLE_EXPOSITION = 1
    ARABLE_EXPOSITION_DELTA = 2
    ICE_SRC = 3
    ICE_ADJACENT = 4
    ICE_DIV = 5
    ORE_SRC = 6
    ORE_DIV = 7
    FACTORY_SEPARATION = 8
    N_CLEAR_CONNECTIONS = 9
    BOARD_MIDDLENESS = 10
    N_FACTORS = 11


class PlantManager:

    # Singleton pattern
    instance = None

    @staticmethod
    def get_instance() -> "PlantManager":
        assert PlantManager.instance is not None
        return PlantManager.instance

    @staticmethod
    def new_instance(player, env_cfg, genome):
        PlantManager.instance = PlantManager(player, env_cfg, genome)
        return PlantManager.instance

    def __init__(self, player, env_cfg, genome):
        self.player = player
        self.opp_player = utils.opp_player(player)
        self.env_cfg = env_cfg
        self.genome = genome

        # Our plants
        self.our_plants = {}
        self.opp_plants = {}

        self.our_lichen_prev = None
        self.our_lichen_current = np.zeros((env_cfg.map_size, env_cfg.map_size), dtype=np.int32)

        # Arable stats
        self.factory_dominance_map = {}     # {fid: binary map where the factory strain is dominant}
        self.primary_ice_neighborhood_map = None
        # todo, maintain the list mining costs for all neighboring resources

    def update_plants(self, game_state):
        """ Called every step. """
        factory_ice_sources = board_routines.get_ice_sources(game_state)

        for fid, factory in game_state.factories[self.player].items():
            if fid not in self.our_plants:
                self.our_plants[fid] = Plant(factory)
            else:
                self.our_plants[fid].update_factory(factory)
            self.our_plants[fid].primary_ice_loc = factory_ice_sources.get(fid, None)

        for fid, factory in game_state.factories[self.opp_player].items():
            if fid not in self.opp_plants:
                self.opp_plants[fid] = Plant(factory)
            else:
                self.opp_plants[fid].update_factory(factory)

        # Retire destroyed factories
        for fid, plant in list(self.our_plants.items()):
            if fid not in game_state.factories[self.player]:
                plant.retire()
                self.our_plants.pop(fid, None)

        for fid, plant in list(self.opp_plants.items()):
            if fid not in game_state.factories[self.opp_player]:
                plant.retire()
                self.opp_plants.pop(fid, None)

        self.our_lichen_prev = np.copy(self.our_lichen_current)
        self.our_lichen_current = board_routines.get_lichen_map(game_state, self.player)

    def get_plant(self, fid):
        return self.our_plants.get(fid, None)

    def get_primary_ice_neighborhood_map(self, game_state):
        if self.primary_ice_neighborhood_map is None:
            pinm = np.zeros_like(game_state.board.ice)
            for fid, f in self.our_plants.items():
                pinm[f.primary_ice_loc] = 1
            pinm = np.clip(convolve2d(pinm, np.ones((3, 3)), mode='same'), 0, 1)
            self.primary_ice_neighborhood_map = pinm
        return self.primary_ice_neighborhood_map

    def factory_cross_distance_mx(self, game_state):
        dd = {}
        for our_fid, our_f in self.our_plants.items():
            for opp_fid, opp_f in self.opp_plants.items():
                # todo, use real cost? or maybe just an approximation is ok
                dd.setdefault(our_fid, {})[opp_fid] = utils.manhattan_distance(our_f.loc, opp_f.loc) / 90

        lp = {}
        for our_fid, our_f in self.our_plants.items():
            lp[our_fid] = our_f.get_lichen_potential(game_state)

        for opp_fid, opp_f in self.opp_plants.items():
            lp[opp_fid] = opp_f.get_lichen_potential(game_state)

        return dd, lp

    def get_our_lichen_under_attack_map(self):
        return np.clip((self.our_lichen_prev - self.our_lichen_current - 1), 0, 1)

    def calculate_strain_dominance_map_(self, game_state):
        """ Simulate lichen growth from the current situation. Private use. """

        x_rubble_ls = utils.get_rubble_lichen_spread(game_state)
        # x_rubble_ls: 0 -- clear tile, 1 -- lichen cannot grow

        # Lichen strain map, -1 means no initial strain, actual strain id start at 0
        lichen_strain = np.copy(game_state.board.lichen_strains)

        # In lichen_dist -1 means rubble -- cannot grow nor spread, 1000 is a starting distance/background value
        lichen_dist = np.where(x_rubble_ls > 0, -1, 1000)

        # Initialize growth from the factories through their connection tiles
        for plant in {**self.our_plants, **self.opp_plants}.values():
            assert isinstance(plant, Plant)
            fct = plant.get_connection_tiles()
            lichen_strain[tuple(fct.T)] = np.where(x_rubble_ls[tuple(fct.T)] == 0, plant.strain_id, -1)
            assert plant.strain_id == int(plant.unit_id.split('_')[1])

        # Set initial distance 0 wherever lichen is already growing or where it has been just seeded
        lichen_dist[lichen_strain > -1] = 0

        # Simulate growth of lichen
        m = game_state.env_cfg.map_size
        wild_strains = (np.arange(0, m*m) + 100).reshape(m, m)
        lsw = np.where((lichen_strain == -1) & (x_rubble_ls == 0), wild_strains, lichen_strain)
        ldx, lsx = board_routines.simulate_lichen_growth_wild(lichen_dist, lsw)

        strain_dominance_map = lsx
        return strain_dominance_map

    def update_arable_stats(self, game_state):
        """ Update is called periodically by the agent. """

        all_plants = {**self.our_plants, **self.opp_plants}

        lsx = self.calculate_strain_dominance_map_(game_state)

        self.factory_dominance_map = {plant.id: np.where(lsx == plant.strain_id, 1, 0) for
                                      plant in all_plants.values()}

        # 1. Number of tiles under dominance (i.e. == strain_area)
        for fid, plant in self.our_plants.items():
            plant.dominance_area = np.sum(lsx == plant.strain_id)

            # Utilization is the percentage of our dominance tiles lichen actually grows
            utilization = np.sum(game_state.board.lichen_strains == plant.strain_id) / (plant.dominance_area + 1)
            plant.current_utilization = max(utilization, utils.low_pass(plant.dominance_area, 7, 15))

    def get_dominance_map(self, fid):
        return self.factory_dominance_map[fid]

    @staticmethod
    @cached(cache=LRUCache(maxsize=1), key=lambda game_state: hashkey(game_state.episode_id, game_state.real_env_steps//10))
    def get_grazing_exclusion_map(game_state):
        # Exclude tiles belonging to opponent's territory
        _, opp_tm = board_routines.get_territory_maps(game_state)

        # Exclude opponent's territory and resources
        grazing_exclusion_map = np.clip(opp_tm + game_state.board.ice + game_state.board.ore, 0, 1)

        return grazing_exclusion_map

    def estimate_water_usage(self, game_state):
        step = game_state.real_env_steps
        all_plants = {**self.our_plants, **self.opp_plants}

        assert game_state.env_cfg.MIN_LICHEN_TO_SPREAD == 20
        assert game_state.env_cfg.LICHEN_WATERING_COST_FACTOR == 10

        nhoodmx = NhoodMatrix.get_instance(game_state.board.rubble.shape)

        x_rubble_ls = utils.get_rubble_lichen_spread(game_state)

        # Lichen strain map, -1 means no initial strain, actual strain id start at 0
        lichen_strain = np.copy(game_state.board.lichen_strains)
        lichen_amount = np.copy(game_state.board.lichen)

        # -1 means rubble -- lichen cannot grow nor spread
        lichen_amount[x_rubble_ls > 0] = -1

        # Initialize growth from the factories through their connection tiles
        for plant in all_plants.values():
            assert isinstance(plant, Plant)
            fct = plant.get_connection_tiles()

            lichen_strain[tuple(fct.T)] = np.where(lichen_amount[tuple(fct.T)] == 0, plant.strain_id, lichen_strain[tuple(fct.T)])
            lichen_amount[tuple(fct.T)] = np.where(lichen_amount[tuple(fct.T)] == 0, 1, lichen_amount[tuple(fct.T)])
            assert plant.strain_id == int(plant.unit_id.split('_')[1])

        # Assume we start watering only after step 800
        water_usage = {}
        for k in range(step, 1000, game_state.env_cfg.LICHEN_WATERING_COST_FACTOR):
            # Sum up water usage for each factory
            for fid, plant in all_plants.items():
                water_usage[fid] = water_usage.get(fid, 0) + min(49, np.sum(lichen_strain == plant.strain_id)) + \
                                   game_state.env_cfg.LICHEN_WATERING_COST_FACTOR * game_state.env_cfg.FACTORY_WATER_CONSUMPTION

            # Grow the existing lichen by 10 (game_state.env_cfg.LICHEN_WATERING_COST_FACTOR)
            lichen_amount += 10 * np.clip(lichen_amount, 0, 1)

            # Spread, assuming watering action always on
            cells = set(map(tuple, np.argwhere(lichen_strain > -1)))
            for cell in cells:
                for neighbor in nhoodmx.n4[cell]:
                    # Min lichen to spread is 20, so it's like: MIN_LICHEN_TO_SPREAD / LICHEN_WATERING_COST_FACTOR == 2.0
                    if lichen_amount[neighbor] == 0 and 20 <= lichen_amount[cell]:
                        assert lichen_strain[neighbor] == -1
                        lichen_amount[neighbor] = 1
                        lichen_strain[neighbor] = lichen_strain[cell]

        return water_usage

    def plan_watering(self, game_state, home_factory_robots):
        step = game_state.real_env_steps
        actions = {}

        if step > 100 and step % 5 == 0:
            est_water_usage = self.estimate_water_usage(game_state)
            for fid, plant in self.our_plants.items():
                water_usage = est_water_usage[fid]
                water_supply = plant.cargo.water + plant.water_supply_ema * (1000 - step)
                plant.water_balance = water_supply - water_usage

                # todo apply genetic optimization?

                # Do not water if no. lichen tiles exceeds 49 and it's still before 900
                n_lichen_tiles = plant.n_lichen_tiles(game_state)
                blocker = step < 900 and n_lichen_tiles >= 50

                # In the second half of the episode, balance water and power more efficiently
                # Allow early watering only if the plant has at least 3 heavy robots
                allower = plant.water_balance > 1000 - step and \
                          (step > 200 and len(home_factory_robots[10].get(fid, [])) >= 3 or step > 800)

                if plant.cargo.water > 30 and not blocker and allower:
                    actions[fid] = plant.water()
                else:
                    actions[fid] = None
                log.debug(f'{step} {fid}, usage: {water_usage:.2f} vs supply: {water_supply:.2f}, water balance: {plant.water_balance:.2f}')
                plant.prev_action = actions[fid]

        else:
            actions.update({fid: plant.prev_action for fid, plant in self.our_plants.items()})
        return actions

    @cached(cache=LRUCache(maxsize=128), key=lambda self, game_state, robot, home_factory_robots:
            hashkey((game_state.ep_step_hash, robot.id)))
    def find_new_home(self, game_state, robot, home_factory_robots) -> (str, float):
        """ Find a plant which would need the robot most, take distance into account. """

        scores = {}
        for fid, plant in self.our_plants.items():
            n_robots_at_plant = len(home_factory_robots[robot.weight].get(fid, []))
            scores[fid] = plant.robot_transfer_score(game_state, robot, n_robots_at_plant)

        # The winner is the plant with the highest score/count proportion
        winner_fid = max(scores.keys(), key=lambda k: scores[k])
        # Transfer score is the difference between the winner and the current home
        # beware, the current robot.home_factory_id may not exist anymore
        score = scores[winner_fid] - scores.get(robot.home_factory_id, 0)
        return winner_fid, score

    def early_setup(self, game_state, remainingOverageTime: int = 60):

        self.update_plants(game_state)

        if game_state.env_steps == 0:
            # Use this step to do early recon, how many good spots there are?
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=np.random.choice([21, 23, 31, 33]) if self.genome.flags['initial_bid'] else 0)
            # return dict(faction="AlphaStrike", bid=0)

        else:

            # It may be our turn to place a factory, but may be not, and still we may do some usefull computation?

            # How much water and metal you have in your starting pool to give to new factories?
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # How many factories you have left to place?
            factories_to_place = game_state.teams[self.player].factories_to_place
            # Whether it is your turn to place a factory?
            my_turn_to_place = utils.my_turn_to_place_factory(game_state.teams[self.player].place_first, game_state.env_steps)

            if not (factories_to_place > 0 and my_turn_to_place):
                log.debug('It\'s not our turn or no more factories to place..')
                return dict()

            else:
                # That's our turn to place a factory

                M = game_state.env_cfg.map_size
                score = np.zeros((*(M, M), ScoreLayers.N_FACTORS))

                # Factory occupancy map, just binary (players indistinguishable)
                fom = np.where(game_state.board.factory_occupancy_map > -1, 1, 0)

                # Devaluate ice fields that are main sources of already placed factories
                factory_ice_sources, _ = board_routines.assign_ice_main_sources(game_state, n_attempts=3)

                # Exclude main ice sources
                x_ice = np.copy(game_state.board.ice)
                for fid, loc in factory_ice_sources.items():
                    x_ice[loc] = 0

                # We don't exlude "main" ore sources, they can be shared
                x_ore = np.copy(game_state.board.ore)

                # We assume digging will be done mainly by heavy robots
                travel_cost_map = utils.get_travel_cost_map(game_state, weight=10)
                travel_cost_map_discounted = travel_cost_map - 1 * game_state.board.rubble // 2
                # We also need the locations of already placed factories
                f_locs = board_routines.get_factory_locs(game_state, game_state.player)

                # Evaluate score for all possible placements
                # ------------------------------------------

                # The new spawn map takes all exclusions into account
                potential_spawns = np.argwhere(game_state.board.valid_spawns_mask == 1)
                for spawn in potential_spawns:
                    # Evaluate ice score
                    max_d = 250
                    ice_loc_dist = board_routines.locate_resources_for_factory(tuple(spawn), x_ice, game_state.board.rubble, travel_cost_map, max_d, rubble_discount=0.25)

                    min_d = ice_loc_dist[0][1] if ice_loc_dist else max_d

                    # So maximum score is when bare ice is just next to the factory, which yields score 150..
                    # when ice is 1 tile away, and with rubble 20+20, the score would be 150 * (20/60) = 50

                    main_source_score = 4500 / np.clip(min_d, 20, 250) - 19
                    score[(*spawn, ScoreLayers.ICE_SRC)] = main_source_score if main_source_score > 0.0 else -300

                    # If the ICE_SRC < 0, save RemainingOverage and continue
                    if score[(*spawn, ScoreLayers.ICE_SRC)] < 0.0:
                        continue

                    # Evaluate ore score
                    max_d = 500

                    ore_loc_dist = board_routines.locate_resources_for_factory(tuple(spawn), x_ore, game_state.board.rubble, travel_cost_map_discounted, max_d, rubble_discount=0.25)
                    min_d = ore_loc_dist[0][1] if ore_loc_dist else max_d
                    assert max_d >= min_d > 0
                    main_ore_source_score = ((max_d - min_d) / max_d) ** 2 * 170
                    score[(*spawn, ScoreLayers.ORE_SRC)] = main_ore_source_score

                    # Add bonus if ICE is adjacent to the potential factory
                    fct = Plant.Get_connection_tiles(spawn, M)
                    ice_adjacent = any([x_ice[tuple(loc)] for loc in fct])
                    score[(*spawn, ScoreLayers.ICE_ADJACENT)] = 30 if ice_adjacent else 0

                    # Number of ice sources is also a factor, it actually can be a big factor for light robot-oriented
                    # strategy
                    ice_diversity_score = np.sum(np.clip((10 - board_routines.DistanceMatrix.get_instance(spawn)) * x_ice, 0, 10) / 10)
                    # Put a cap on ice diversity, a single factory won't be able to utilize more
                    score[(*spawn, ScoreLayers.ICE_DIV)] = np.tanh(ice_diversity_score / 2) * 50

                    # Nearby ore score
                    ore_diversity_score = np.sum(np.clip((20 - board_routines.DistanceMatrix.get_instance(spawn)) * x_ore, 0, 20) / 20)
                    # A cap on ore diversity, a single factory won't be able to utilize more
                    score[(*spawn, ScoreLayers.ORE_DIV)] = np.tanh(ore_diversity_score / 2) * 50

                    # Factory separation score
                    distances = utils.manhattan_distances(spawn, f_locs)
                    d_score = (min(distances) + sum(distances) / len(distances)) if f_locs else 0
                    score[(*spawn, ScoreLayers.FACTORY_SEPARATION)] = 0.5 * d_score

                # Take into account our existing factories overlap
                arable_map = (1 - game_state.board.rubble/100)**1.5 * (1 - game_state.board.ice) * (1 - game_state.board.ore) * (1 - fom)
                our_placed_factories_dmx = np.zeros_like(arable_map)
                for plant in self.our_plants.values():
                    dmx = np.clip(10 - board_routines.DistanceMatrix.get_instance(plant.loc), 0, 10) / 10
                    our_placed_factories_dmx = np.maximum(our_placed_factories_dmx, dmx)

                # For 50% best Resource locations, calculate the arable score
                score_res = score[:, :, ScoreLayers.ICE_SRC] + score[:, :, ScoreLayers.ICE_DIV] + \
                            score[:, :, ScoreLayers.ORE_SRC] + score[:, :, ScoreLayers.ORE_DIV]
                p50 = np.percentile(score_res, 50)
                # When p50 is negative, the algorithm might consider invalid locations..
                # make sure we examine only locations that have positive preliminary score
                for spawn in np.argwhere(score_res > max(1.0, p50)):

                    dmx = np.clip(10 - board_routines.DistanceMatrix.get_instance(tuple(spawn)), 0, 10) / 10
                    dmx[spawn[0] - 1:spawn[0] + 2, spawn[1] - 1:spawn[1] + 2] = 0
                    arable_exposition = 1.0 * np.sum(dmx * arable_map)

                    # Subtract the area we gonna take from our other factories, however sharing a big lake is not a
                    # bad idea, as it makes it easier, 1 factory can accumulate about 1500 water, and it's enough to
                    # fully grow about 100 tiles

                    # Absolute maximum of arable exposition is 90
                    arable_exposition_delta = 2.0 * (np.sum(np.maximum(our_placed_factories_dmx, dmx) * arable_map -
                                                     our_placed_factories_dmx * arable_map))

                    # Lichen can be watered/connected only through the connection/adjacent tiles on all sides of a
                    # factory, having a few connection tiles from the start is good, but not critical anymore.
                    fct = Plant.Get_connection_tiles(spawn, M)
                    n_clear_connection_tiles = 2 * np.sum(game_state.board.rubble[tuple(fct.T)] == 0)

                    # todo try to reduce arable exposition, and rely on delta more..
                    score[(*spawn, ScoreLayers.ARABLE_EXPOSITION)] = arable_exposition
                    score[(*spawn, ScoreLayers.ARABLE_EXPOSITION_DELTA)] = arable_exposition_delta
                    score[(*spawn, ScoreLayers.N_CLEAR_CONNECTIONS)] = n_clear_connection_tiles
                    score[(*spawn, ScoreLayers.BOARD_MIDDLENESS)] = 50 * (1 - np.sqrt((spawn[0] - M//2)**2 + (spawn[1] - M//2)**2) / (24 * np.sqrt(2)))

                total_score = np.zeros_like(score[:, :, 0])
                for key, w in self.genome.placement_weights.items():
                    total_score += w * score[:, :, ScoreLayers[key]]

                best_loc = np.unravel_index(np.argmax(total_score), total_score.shape)

                score_dict = {sl.name: np.round(score[best_loc[0], best_loc[1], sl.value], 2) for sl in ScoreLayers if sl.value < ScoreLayers.N_FACTORS}

                log.info(f'{self.player}: best location found: {best_loc}, score: {score_dict} -> {total_score[best_loc]:.2f}')

                if hasattr(log, 'SHOW_PLOTS') and log.SHOW_PLOTS and self.player == 'player_0':
                    # Just for visualization
                    score[:, :, ScoreLayers.RUBBLE] = game_state.board.rubble
                    board_routines.plot_scores(score, [ScoreLayers.RUBBLE,
                                                       ScoreLayers.ICE_SRC,
                                                       ScoreLayers.ICE_DIV,
                                                       ScoreLayers.ORE_DIV,
                                                       ScoreLayers.FACTORY_SEPARATION,
                                                       ScoreLayers.ARABLE_EXPOSITION,
                                                       ScoreLayers.ARABLE_EXPOSITION_DELTA,
                                                       ScoreLayers.N_CLEAR_CONNECTIONS,
                                                       ScoreLayers.BOARD_MIDDLENESS,
                                                       ], [])
                    plt.show()

                return dict(spawn=list(best_loc), metal=min(metal_left, 150), water=min(water_left, 150))

