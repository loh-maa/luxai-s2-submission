from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import numpy as np

from a09_3a import utils, path_finding
from a09_3a import board_routines
from a09_3a.board_routines import DistanceMatrix, BoardManager
from a09_3a.logger import log

from lux.factory import Factory


class Plant:

    def __init__(self, factory: Factory):
        self.factory = factory
        self.id = self.factory.unit_id

        self.loc = tuple(self.factory.pos)
        self.factory_tiles_ = None
        self.connection_tiles_ = None

        # Dynamic variables
        self.prev_action = None
        self.primary_ice_loc = None

        self.dominance_area = None

        # Current utilization of the arable area -- how many lichen tiles / connected area
        self.current_utilization = None
        self.emergency_water_robot_assigned = False

        self.water_balance = 0
        self.water_supply_ema = 0.0
        self.wait_for_metal = -100

        # 10 nearest resource tiles, sorted by L distance
        self.nearest_ice_tiles = None
        self.nearest_ore_tiles = None
        self.ice_diversity = None
        self.ore_diversity = None

    def __getattr__(self, attr):
        # If the attr was not found in Plant, try to find it in the factory
        return getattr(self.factory, attr)

    def update_factory(self, factory: Factory):
        water_delta = factory.cargo.water - self.factory.cargo.water
        metal_delta = factory.cargo.metal - self.factory.cargo.metal

        # Wait for metal countdown.. if no new metal is available, the factory will produce light robots
        self.wait_for_metal = 10 if metal_delta > 0 else self.wait_for_metal - 1

        self.factory = factory
        water_supply = max(0.0, water_delta)
        self.water_supply_ema = 0.99 * self.water_supply_ema + 0.01 * water_supply

    def retire(self):
        # Clean up the plant from registries
        pass

    def is_factory_tile(self, loc):
        return abs(self.loc[0] - loc[0]) <= 1 and abs(self.loc[1] - loc[1]) <= 1

    def get_factory_tiles(self):
        """ Return coordinates of all the 9 cells making up the factory. """
        if self.factory_tiles_ is None:
            self.factory_tiles_ = utils.get_factory_tiles(self.loc)
        return self.factory_tiles_

    @staticmethod
    def Get_connection_tiles(loc, map_size):
        """ Return coordinates of all the cells neighboring with a factory located at loc, which is 12x2. """
        # i = [-2, 0, 2, 0]
        # j = [0, 2, 0, -2]
        i = [-2, -2, -2, -1, 0, 1, 2, 2, 2, 1, 0, -1]
        j = [-1, 0, 1, 2, 2, 2, 1, 0, -1, -2, -2, -2]
        frame = np.array([i, j]) + np.array(loc).reshape(2, 1)
        # Remove out of bounds cells
        frame = [tuple(ij) for ij in frame.T if utils.is_in_bounds(ij, map_size)]
        connection_tiles = np.array(frame)
        assert connection_tiles.shape[1] == 2
        return connection_tiles

    def get_connection_tiles(self):
        """ Return coordinates of all the 4 cells connecting the factory with lichen areas. """
        if self.connection_tiles_ is None:
            self.connection_tiles_ = Plant.Get_connection_tiles(self.loc, self.factory.env_cfg.map_size)
        return self.connection_tiles_

    def robot_transfer_score(self, game_state, robot, n_robots_at_plant):
        """ Calculate how good it would be to have a new robot (but it has to travel) """
        step = game_state.real_env_steps
        dmx = np.clip(10 - DistanceMatrix.get_instance(self.loc), 0, 10) / 10
        l_dist = utils.manhattan_distance(robot.loc, self.loc)

        score = (20 * utils.hi_pass(self.power, 3000, 13000) +
                7 * utils.hi_pass(self.dominance_area, 0, 140) +
                7 * utils.hi_pass(np.sum(dmx * (game_state.board.ice + game_state.board.ore)), 0, 10))
        # Take robot counts into account, if the robot belongs to this plant, count it out
        belongs = int(robot.home_factory_id == self.id)
        if robot.weight == 1:
            z = (10 - (1 + n_robots_at_plant - belongs))
            score += np.sign(z) * abs(z) ** 1.5
        else:
            z = (4 - (1 + n_robots_at_plant - belongs))
            score += np.sign(z) * abs(z) ** 2
        # todo Travel penalty? try out 0.1
        score -= 0.15 * l_dist
        # Timeline window, transfers make sense in the middle game
        # Do not take step into account, coz the results can be weird, for instance if all factories have many robots
        # the scores are negative, but for distant factory at the end of the episode the score would zero-out
        # and there would be a big score different...
        # score *= utils.hi_pass(step, 50, 200) * utils.low_pass(step, 850 - 3 * l_dist, 950 - 3 * l_dist)
        return score

    def get_nearest_ice_tiles(self):
        if self.nearest_ice_tiles is None:
            ice_locs = BoardManager.get_instance().ice_locs
            ice_dist = list(zip(ice_locs, utils.manhattan_distances(self.loc, ice_locs)))
            self.nearest_ice_tiles = sorted(ice_dist, key=lambda x: x[1])
        return self.nearest_ice_tiles

    def get_nearest_ore_tiles(self):
        if self.nearest_ore_tiles is None:
            ore_locs = BoardManager.get_instance().ore_locs
            ore_dist = list(zip(ore_locs, utils.manhattan_distances(self.loc, ore_locs)))
            self.nearest_ore_tiles = sorted(ore_dist, key=lambda x: x[1])
        return self.nearest_ore_tiles

    def get_ice_diversity(self):
        """ 1.0 means abundance, 0.0 not even 1 src, 0.1 < x < 0.2 just a solid 1 src. """
        if self.ice_diversity is None:
            nit = self.get_nearest_ice_tiles()
            ice_div = sum([max(0, 10 - d) for it, d in nit]) / 70
            self.ice_diversity = min(ice_div, 1.0)
            log.debug(f'{self.id} ice diversity: {self.ice_diversity}')
        return self.ice_diversity

    def get_ore_diversity(self):
        """ 1.0 means abundance, 0.0 not even 1 src, 0.1 < x < 0.2 just a solid 1 src. """
        if self.ore_diversity is None:
            nit = self.get_nearest_ore_tiles()
            ore_div = sum([max(0, 15 - d) for it, d in nit]) / 100
            self.ore_diversity = min(ore_div, 1.0)
            log.debug(f'{self.id} ore diversity: {self.ore_diversity}')
        return self.ore_diversity

    @cached(cache=LRUCache(maxsize=8), key=lambda self, game_state: hashkey(self.id, game_state.ep_step_hash))
    def get_lichen_map(self, game_state):
        return np.where(game_state.board.lichen_strains == self.strain_id, game_state.board.lichen, 0)

    @cached(cache=LRUCache(maxsize=8), key=lambda self, game_state: hashkey(self.id, game_state.ep_step_hash))
    def n_lichen_tiles(self, game_state):
        return np.sum(game_state.board.lichen_strains == self.strain_id)

    def needs_emergency_water(self):
        return self.factory.cargo.water < 50

    @cached(cache=LRUCache(maxsize=16), key=lambda self, game_state, weight, resource:
            hashkey(self.id, game_state.episode_id, game_state.real_env_steps//10, weight, resource))
    def find_best_resources(self, game_state, weight, resource='ice'):
        """ Calculate the cost of going to the nearest/most available resource tile, uncovering it from rubble
        (discounted in early stage) and bringing the resource back to the factory. """

        drr = game_state.env_cfg.ROBOTS['LIGHT' if weight == 1 else 'HEAVY'].DIG_RUBBLE_REMOVED
        dc = game_state.env_cfg.ROBOTS['LIGHT' if weight == 1 else 'HEAVY'].DIG_COST

        # If the tile is rubbled and in the opponent's territory, make it more expensive
        tadv = board_routines.get_territorial_advantage(game_state)

        distance_map = utils.get_travel_cost_map(game_state, weight=weight)
        can_loc_dist = self.get_nearest_ice_tiles() if resource == 'ice' else self.get_nearest_ore_tiles()
        # The locs are sorted by L distance, take at most N nearest
        cost_loc = []
        for res_loc, dist in can_loc_dist[:15]:
            assert isinstance(res_loc, tuple)
            _, leg_1_cost, _ = path_finding.find_shortest_path_directions(distance_map, self.loc, [res_loc])
            _, leg_2_cost, _ = path_finding.find_shortest_path_directions(distance_map, res_loc, self.get_factory_tiles())
            n_digs = int(np.ceil(game_state.board.rubble[res_loc] / drr))

            opp_ter_penalty = 1 - min(0, tadv[res_loc]) * (1 + game_state.board.rubble[res_loc]/30) * ((1000 - game_state.real_env_steps)/500)
            cost = sum(leg_1_cost) + sum(leg_2_cost) + n_digs * dc * ((200 + game_state.real_env_steps)/500) * opp_ter_penalty
            cost_loc.append((cost, res_loc))

        # Return a list of (cost, loc) tuples, sorted by cost
        cost_loc = sorted(cost_loc, key=lambda x: x[0])
        return cost_loc

    def under_attack(self, game_state):
        # Quick estimation based on the average risk around the factory..
        sr = utils.slice_radius(self.loc, r=5)
        risk_map = board_routines.get_risk_map(game_state)[sr] / 40.0
        lichen_map = self.get_lichen_map(game_state)[sr] / 100.0
        dmx = (5 - DistanceMatrix.get_instance(self.loc)[sr]) / 5.0

        under_attack = np.sum(risk_map * lichen_map * dmx)
        if under_attack > 1.0:
            log.debug(f'{self.id} is under attack')
        return under_attack

    @cached(cache=LRUCache(maxsize=8), key=lambda self, game_state: hashkey(self.id, game_state.ep_step_hash))
    def get_lichen_potential(self, game_state):
        """ How much lichen this factory is growing or is capable of growing at the end, in [0, 1]. """

        step = game_state.real_env_steps

        other_lichen = ((game_state.board.lichen_strains >= 0) * (
                    game_state.board.lichen_strains != self.factory.strain_id)).astype(np.int)

        nlt = max(1, self.n_lichen_tiles(game_state))
        rls = 1 - np.clip((utils.get_rubble_lichen_spread(game_state) + other_lichen), 0, 1)
        dmx = np.clip(1 - DistanceMatrix.get_instance(self.loc) / 10, 0, 1)
        arability = np.sum(rls * dmx ** 2) / np.sum(dmx ** 2)
        lp, nt20 = board_routines.lichen_projection(step, n_tiles0=nlt, water=self.factory.cargo.water,
                                                    arability=arability)
        lpl = round(np.clip(lp / 15000, 0.0, 1.0), 3)
        log.debug(f'@{step} {self.id} lichen potential: {lp:.3f}, limited: {lpl}')
        return lpl

    def has_plenty_of_power(self, game_state, for_robot):
        thold = 4000 + for_robot.battery - for_robot.power - 2000 * utils.hi_pass(game_state.real_env_steps, 800, 900)
        return self.factory.power > thold
