from cachetools import cached, LRUCache
from cachetools.keys import hashkey
import copy
from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from a09_3a import board_routines
from a09_3a import path_finding
from a09_3a import utils
from a09_3a.board_routines import BoardManager
from a09_3a.genome import Genome
from a09_3a.plant import Plant
from a09_3a.plant_manager import PlantManager
from a09_3a.logger import log

from lux.unit import Unit


class MissionOnTop(Exception):

    def __init__(self, new_mission, **kwargs):
        super().__init__()
        self.new_mission = new_mission
        self.kwargs = kwargs


class Mission(Enum):
    # No mission assigned
    NONE = 0
    TRANSFER = 1
    RECHARGE = 2
    GO_ICE = 3
    GO_ORE = 4
    GO_HOME = 5
    GUARD = 6
    GRAZING = 7
    HUNT = 8
    SABOTAGE = 9
    BRING_WATER = 10
    ATTACK = 11
    KAMIKAZE = 12
    SUPPORT = 13
    WAIT = 14
    SOLAR = 15
    # Not really a mission, just to use the score tab
    POWER_PICKUP = 16
    TEST = 17


class MissionScoreTab:

    def __init__(self):
        self.tab = {}

    def get(self, mission: Mission) -> float:
        return self.tab.get(mission, [0])[0]

    def set(self, mission: Mission, score: float, turns: int) -> None:
        self.tab[mission] = [score, turns]

    def countdown(self) -> None:
        for k, v in self.tab.items():
            v[1] = max(0, v[1]-1)
            if v[1] == 0:
                v[0] = 0


class Robot:
    """ High level wrapper around lux.unit.Unit """

    # Unit fields:
    #
    # action_queue: List
    # cargo: UnitCargo
    # env_cfg: EnvConfig
    # pos: np.ndarray
    # power: int
    # team_id: int
    # unit_cfg: dict
    # unit_id: str
    # unit_type: str # "LIGHT" or "HEAVY"

    next_locs = {}      # {uid: loc}

    # Includes ice, ore, grazing, sabotage destinations... not for hunting though..
    dst_locs = {}       # {uid: loc}
    hvy_dst_locs = {}  # {uid: loc}
    preys_targeted = {} # {uid: uid}
    emergency_water_robots = {} # {uid: fid}

    # Stats
    StatsTemplate = {
        'hunts_completed': 0,
        'light_units_made': 0,
        'light_units_lost': 0,
        'heavy_units_made': 0,
        'heavy_units_lost': 0,
        'water_brought': 0,
        'water_toolate': 0,
        'missions': {
            1: defaultdict(int),
            10: defaultdict(int)
        }
    }
    stats = {}

    chain_maxlen = 4

    def __init__(self, unit: Unit, home_factory_id: str, genome: Genome):
        self.unit = unit
        self.id = self.unit.unit_id
        self.genome = genome
        # Static variables
        self.weight = 10 if self.unit.unit_type == 'HEAVY' else 1
        self.battery = unit.unit_cfg.BATTERY_CAPACITY
        self.avg_sabotage_cost = 1.8 * self.weight

        self.action_queue_cost = self.weight
        self.dc = unit.unit_cfg.DIG_COST

        # The primary heavy robot may have some priority, mostly static but can change
        # self.is_primary_heavy = False
        self.is_primary_ice = False
        self.is_primary_ore = False

        # Stats
        if self.is_heavy():
            Robot.stats['heavy_units_made'] += 1
        else:
            Robot.stats['light_units_made'] += 1

        # Dynamic variables, should be updated frequently or every turn
        self.loc = tuple(self.unit.pos)
        self.pweight = unit.get_power_weight()
        assert home_factory_id
        self.home_factory_id = home_factory_id
        self.mission = Mission.NONE

        # Keep score modifiers here, e.g. when the agent selects mission ICE but it turns out the ice locations are
        # too far or busy, it may set a temporary score modifier to avoid repeating the ICE mission choice
        # {mission: [score_mod, turns_remaining]}
        self.score_tab = MissionScoreTab()
        self.target_factory_id = None

        self.chain_path = None
        self.power_needed_per_step = 39

    def __str__(self) -> str:
        return self.id

    def __getattr__(self, attr):
        # If the attr was not found in Robot, try to find it in the unit
        return getattr(self.unit, attr)

    @classmethod
    def reset(cls):
        Robot.next_locs = {}
        Robot.dst_locs = {}
        Robot.hvy_dst_locs = {}
        Robot.preys_targeted = {}
        Robot.emergency_water_robots = {}
        # Stats
        Robot.stats = copy.deepcopy(Robot.StatsTemplate)

    def update_unit(self, unit: Unit):
        """ Called every turn to update the underlying unit object. """

        # We can collect stats here, based on difference between the prev and current unit state
        self.unit = unit
        self.loc = tuple(self.unit.pos)
        self.score_tab.countdown()
        self.pweight = unit.get_power_weight()
        if self.id in Robot.dst_locs and self.is_heavy():
            Robot.hvy_dst_locs[self.id] = Robot.dst_locs[self.id]
        else:
            Robot.hvy_dst_locs.pop(self.id, None)

    def retire(self, game_state):
        # Clean up the robot from registries
        Robot.next_locs.pop(self.id, None)
        Robot.dst_locs.pop(self.id, None)
        Robot.hvy_dst_locs.pop(self.id, None)
        Robot.preys_targeted.pop(self.id, None)
        Robot.emergency_water_robots.pop(self.id, None)

        # Count the units lost only before step 900
        if game_state.real_env_steps < 900:
            if self.is_light():
                Robot.stats['light_units_lost'] += 1
            elif self.is_heavy():
                Robot.stats['heavy_units_lost'] += 1

    def is_idle(self):
        return len(self.unit.action_queue) == 0

    def is_heavy(self):
        return self.weight == 10

    def is_light(self):
        return self.weight == 1

    def power2b(self):
        return self.unit.power / self.battery

    def is_aggressive(self):
        return self.is_mission_any([Mission.ATTACK, Mission.GRAZING, Mission.GUARD, Mission.HUNT, Mission.SABOTAGE])

    def get_mission(self):
        return self.mission

    def is_mission(self, mission):
        return self.mission == mission

    def is_mission_any(self, missions):
        return self.mission in missions

    def set_mission(self, mission):
        """ Setting a new mission, wipes out all the current variables related to the current mission. """
        log.debug(f'{self.id} new mission: {self.mission} -> {mission}')

        if self.mission == Mission.BRING_WATER:
            Robot.emergency_water_robots.pop(self.id, None)

        if self.mission == Mission.HUNT:
            Robot.preys_targeted.pop(self.id, None)

        self.mission = mission
        Robot.dst_locs.pop(self.id, None)
        Robot.hvy_dst_locs.pop(self.id, None)
        self.chain_path = None

    def mission_accomplished(self, mission: Mission = None):
        assert mission is None or mission == self.mission
        log.debug(f'{self.id}: {self.mission} mission accomplished')
        # Set the NONE mission through, set_mission()
        self.set_mission(Mission.NONE)

    def has_mission_critical(self):
        return self.mission == Mission.BRING_WATER

    @staticmethod
    def is_digging(unit):
        return len(unit.action_queue) > 0 and unit.action_queue[0][0] == 3

    def is_harassed(self, game_state):
        return board_routines.get_risk_map(game_state)[self.loc] >= self.weight

    def is_next_action_equiv(self, al: list):
        """ Check whether the next actions in both queues are equivalent. """

        # If both are empty, return true, else if either is empty return false
        if len(al) == 0 and len(self.unit.action_queue) == 0:
            return True
        # Check for RECHARGE == empty list equivalence (MOVE CENTER is also equvalent, but we never use it)
        elif len(self.unit.action_queue) == 0 and len(al) > 0 and al[0][0] == 5 or \
                len(al) == 0 and len(self.unit.action_queue) > 0 and self.unit.action_queue[0][0] == 5:
            return True
        elif len(al) == 0 or len(self.unit.action_queue) == 0:
            return False

        assert isinstance(al[0], np.ndarray) and len(al[0]) > 2

        a = self.unit.action_queue[0]
        b = al[0]
        return a[0] == b[0] == 0 and a[1] == b[1] or \
            a[0] == b[0] == 1 and a[1] == b[1] and a[2] == b[2] or \
            a[0] == b[0] == 2 and a[2] == b[2] or \
            a[0] == b[0] and b[0] > 2

    @staticmethod
    def calculate_next_location(unit, game_state, new_actions=None):
        """ We want to apply this method to the opponent units (which are not wrapped with Robot) as well, thus
        it is static and takes unit as argument. """

        aq_to_be_updated = False
        # If we plan to update the action queue and we actually have enough power to update the queue...
        aqc = unit.action_queue_cost(game_state)
        if new_actions is not None and unit.power >= aqc:
            aq = new_actions
            aq_to_be_updated = True
        else:
            aq = unit.action_queue

        if aq is None or len(aq) == 0:
            return unit.loc

        a = aq[0]
        if a[0] == 0:
            # Apply only if we have enough power to move
            if unit.power < unit.move_cost(game_state, a[1]) + (aqc if aq_to_be_updated else 0):
                # The unit will not move, even if MOVE action is in the queue..
                return unit.loc

            if a[1] == 1:
                new_loc = unit.loc[0], unit.loc[1] - 1
            elif a[1] == 2:
                new_loc = unit.loc[0] + 1, unit.loc[1]
            elif a[1] == 3:
                new_loc = unit.loc[0], unit.loc[1] + 1
            elif a[1] == 4:
                new_loc = unit.loc[0] - 1, unit.loc[1]
            else:
                new_loc = unit.loc

            return new_loc if utils.is_in_bounds(new_loc, unit.env_cfg.map_size) else unit.loc

        return unit.loc

    def can_move_onto(self, game_state, loc):
        direction = utils.direction_to(self.loc, loc)
        plan_cost = self.unit.action_queue_cost(game_state)
        if len(self.unit.action_queue) > 0:
            if self.unit.action_queue[0][0] == 0 and self.unit.action_queue[0][1] == direction:
                # The move action already planned!
                plan_cost = 0
        return utils.is_adjacent(self.loc, loc) and self.unit.move_cost(game_state, direction) + plan_cost < self.unit.power

    def estimate_water_delivery_cost(self, dst_plant: Plant):
        """ Calculate the cost of delivering water by this robot from its home_plant. """

        home_plant = PlantManager.get_instance().get_plant(self.home_factory_id)
        if home_plant is None or home_plant.factory.cargo.water < 150:
            return 1000
        wa = 50 * utils.low_pass(home_plant.factory.cargo.water, 150, 400)
        wb = 20 * utils.low_pass(home_plant.water_balance, -100, 0)
        d1 = 2 * utils.manhattan_distance(self.loc, home_plant.loc)
        d2 = 2 * utils.manhattan_distance(home_plant.loc, dst_plant.loc)
        # Prefer non-support robots
        mc = 5 * int(self.is_mission(Mission.SUPPORT))
        if dst_plant.factory.cargo.water < d1//2 + d2//2:
            # This robot won't make it on time anyway
            return 1000
        return wa + wb + d1 + d2 + mc

    def scan_for_prey(self, game_state):
        """ Find a robot within 10 tiles, that has equal weight but little power, too little to return home. """

        bc = self.battery
        oft = board_routines.get_opp_factory_tiles(game_state)

        # Detect class A prey -- the one that cannot escape, likely to be killed with enough pursuit
        for oid, onit in game_state.units[game_state.opp_player].items():
            if onit.weight == self.weight and onit.power / bc < 0.125 and 0.25 < self.power2b():
                l_dist = utils.manhattan_distance(self.loc, onit.loc)
                already_targeted = len([1 for x in Robot.preys_targeted.values() if x == oid])

                # Allow max 2 units attack the same robot..
                if l_dist >= 9 or onit.loc in oft or already_targeted > 2:
                    continue

                # Check if the robot can reach its nearest factory easily or not
                _, cost, _ = self.find_way(game_state, destination_locs=oft, from_loc=onit.loc)
                assert cost
                # Reduce the last step from 1000 to weight (find way treats opponent factory as an obstacle)
                cost[-1] = onit.weight
                cost_solar = utils.modify_cost_by_solar_gain(game_state, step0=game_state.real_env_steps, cost=cost, weight=onit.weight)

                if onit.power + onit.weight * l_dist / 2 < sum(cost_solar):
                    log.debug(f'\t{self.id} found prey: {oid}, cost: {sum(cost)}, cost_solar: {sum(cost_solar)}')
                    return oid

        return None

    # @cached(cache=LRUCache(maxsize=256), key=lambda self, game_state, destination_locs, from_loc=None:
    #         hashkey(self.id, game_state.ep_step_hash, str(destination_locs), from_loc))
    def find_way(self, game_state, destination_locs, from_loc=None, include_our_units=True):
        """ High level path finding -- the shortest path, while avoiding collisions with our units (given the
        anticipated locations in the next step). """
        # step = game_state.real_env_steps

        if isinstance(destination_locs, tuple):
            destination_locs = np.array(destination_locs).reshape(1, 2)
        elif isinstance(destination_locs, np.ndarray):
            assert destination_locs.shape[1] == 2
        elif isinstance(destination_locs, list):
            assert all([isinstance(x, tuple) for x in destination_locs])
        else:
            assert False

        distance_map = utils.get_travel_cost_map(game_state, weight=self.weight)

        # Avoid collisions with our own robots, however assume next locs are affecting only within radius 5 and only
        # of those of equal or heavier units
        if include_our_units:
            distance_map = np.copy(distance_map)
            our_units = game_state.units[game_state.player]
            for uid, loc in Robot.next_locs.items():
                if uid != self.id and self.loc != loc:
                    # This may be not enough to go around a stationary heavy robot
                    distance_map[loc] += round((300 * our_units[uid].weight / self.weight))

        from_loc = from_loc or self.loc
        directions, cost, path = path_finding.find_shortest_path_directions(distance_map, from_loc, destination_locs)
        return directions, cost, path

    @cached(cache=LRUCache(maxsize=256), key=lambda self, game_state: hashkey(self.id, game_state.ep_step_hash))
    def calculate_power_leash(self, game_state):
        our_ft = PlantManager.get_instance().get_plant(self.home_factory_id).get_factory_tiles()
        _, cost, _ = self.find_way(game_state, destination_locs=our_ft, include_our_units=False)

        # todo, genetic param?
        cost_solar = utils.modify_cost_by_solar_gain(game_state, game_state.real_env_steps, cost, self.weight)

        return sum(cost_solar)

    @cached(cache=LRUCache(maxsize=128), key=lambda self, game_state, home_plant, resource:
            hashkey(self.id, game_state.ep_step_hash, home_plant.id, resource))
    def evaluate_mission_go_res(self, game_state, home_plant: Plant, resource: str) -> tuple:
        step = game_state.real_env_steps

        log.debug(f'@{step} {self} {"(primary ice)" if self.is_primary_ice else ""} '
                  f'{"(primary ore)" if self.is_primary_ore else ""}: evaluate mission GO_{resource}')

        # If the tile is rubbled and in the opponent's vinicity, make it more expensive
        tadv = board_routines.get_territorial_advantage(game_state)

        cost_locs = home_plant.find_best_resources(game_state, self.weight, resource=resource)
        primary_ice_loc = home_plant.primary_ice_loc
        # primary_ice_loc may be None!
        # Consider at most 3 available destinations, tuples in cost_locs are already sorted by cost
        scores = []
        for _, dst_loc in cost_locs:
            # Heavy robots have priority
            if self.is_light() and dst_loc in Robot.dst_locs.values() or self.is_heavy() and dst_loc in Robot.hvy_dst_locs.values():
                continue

            # Reserve the primary ice for a heavy robot
            if dst_loc == primary_ice_loc and not self.is_heavy():
                continue

            # Otherwise, find a way to a res tile (preferably the main source of res tile)
            # Note if the path is crowded or the target tile is occupied the cost will be very high, it may also indicate
            # the robot would have to fight for the resource
            _, cost, path_locs = self.find_way(game_state, destination_locs=dst_loc, include_our_units=False)
            cost_go = sum(cost)

            # Estimate the power for the way back
            _, cost, _ = self.find_way(game_state, destination_locs=home_plant.get_factory_tiles(), from_loc=dst_loc,
                                       include_our_units=False)
            cost_back = sum(cost)

            # How many digs we are going to be able to do
            power_available = min(self.battery, self.unit.power + (home_plant.power if home_plant.is_factory_tile(self.loc) else 0))
            n_digs = max(0, (power_available - cost_go - cost_back - self.battery // 10) // self.dc)

            # Efficiency, how much energy we gonna spend on digs vs travel
            efficiency = (n_digs * self.dc) / max(1, self.unit.power)

            # Is the destination area risky?
            risk_map = BoardManager.get_instance().get_risk_map_ema()
            mean_risk = risk_map[dst_loc]

            # Is the resource on the opponent's territory and is rubbled?
            # Opponent vinicity -- tiles very close to opponent's factories are close to 1.0 decreasing outwards
            if resource == 'ore':
                opp_ter_rubble_penalty = min(0, tadv[dst_loc] - 0.25) * (game_state.board.rubble[dst_loc]/50) * \
                                     ((1000 - game_state.real_env_steps)/800)
            else:
                opp_ter_rubble_penalty = 0

            if self.genome.flags['evaluate_go_res_old']:
                # Risk penalty only if going to opponent's territory
                if self.is_light():
                    risk_penalty = min(0.0, tadv[dst_loc] * (mean_risk / self.weight))
                else:
                    risk_penalty = 0
                score = efficiency + risk_penalty + opp_ter_rubble_penalty
            else:
                score = (1.0 - (cost_go + cost_back + 3 * self.unit.action_queue_cost(game_state)) / self.battery) - \
                        0.1 * (mean_risk / self.weight) + opp_ter_rubble_penalty

            scores.append((score, dst_loc))

            log.debug(f'\t... to: {dst_loc}, cost: {cost_go} + {cost_back}, risk: {mean_risk:.2f}, '
                      f'opp_territory_rubble_penalty: {opp_ter_rubble_penalty:.2f}, score: {score:.2f}')

            if len(scores) == 3:
                break

        # We maximize scores
        score, dst_loc = max(scores, default=(0.0, None), key=lambda x: x[0])
        return score, dst_loc

    @cached(cache=LRUCache(maxsize=128), key=lambda self, game_state, home_plant:
            hashkey(self.id, game_state.ep_step_hash, home_plant.id))
    def evaluate_initial_go_ore(self, game_state, home_plant: Plant):
        step = game_state.real_env_steps

        log.debug(f'@{step} {self}: evaluate initial mission GO_ORE')

        # If the tile is rubbled and in the opponent's vinicity, make it more expensive
        tadv = board_routines.get_territorial_advantage(game_state)

        cost_locs = home_plant.find_best_resources(game_state, self.weight, resource='ore')

        # Consider at most 3 available destinations, tuples in cost_locs are already sorted by cost
        scores = []
        for _, dst_loc in cost_locs:
            _, cost, path_locs = self.find_way(game_state, destination_locs=dst_loc, include_our_units=False)
            cost_go = sum(cost)
            # Estimate the power for the way back
            _, cost, _ = self.find_way(game_state, destination_locs=home_plant.get_factory_tiles(), from_loc=dst_loc,
                                       include_our_units=False)
            cost_back = sum(cost)

            # Power needed to bring missing ore
            uncover = int(np.ceil(game_state.board.rubble[dst_loc] / 20)) * self.dc
            ore_needed = (100 - home_plant.cargo.metal) * 5
            power_needed = cost_go + cost_back + uncover + int(np.ceil(ore_needed/20)) * self.dc + 100

            steps_to_wait = int(np.ceil((power_needed - min(3000, self.power + home_plant.power)) / 50))

            opp_ter_rubble_penalty = min(0, tadv[dst_loc]) * (game_state.board.rubble[dst_loc] / 50)

            score = -steps_to_wait + 20 * opp_ter_rubble_penalty
            scores.append((score, dst_loc, steps_to_wait))

            log.debug(f'\t{self}... to: {dst_loc}, cost: {cost_go} + {cost_back}, '
                      f'opp_territory_rubble_penalty: {opp_ter_rubble_penalty:.2f}, score: {score:.2f}')

            if len(scores) == 3:
                break

        # We maximize scores
        score, dst_loc, steps_to_wait = max(scores, default=(0.0, None), key=lambda x: x[0])
        return score, dst_loc, steps_to_wait

    def power_pickup_in_queue(self):
        for a in self.unit.action_queue:
            if a[0] == 2 and a[2] == 4:
                return True
        return False

    def cargo_transfer_in_queue(self):
        for a in self.unit.action_queue:
            if a[0] == 1:
                return True
        return False

    # Chain-related

    def is_chain_needed(self, game_state):
        # What about being harassed.. the chain is automatically cleared when the digging robot is out of the position
        return self.chain_path

    def establish_chain(self, game_state, robots):
        if self.is_harassed(game_state):
            log.debug(f'{self.id} is harassed cannot build a chain')
            self.chain_path = None
            return

        home_plant = PlantManager.get_instance().get_plant(self.home_factory_id)

        # However we must get around any other chains
        other_chains = sum([q.chain_path for q in robots.values() if q.chain_path is not None], [])

        distance_map = np.copy(utils.get_travel_cost_map(game_state, weight=10))
        for tile in other_chains:
            distance_map[tile] = 1000

        _, _, path = path_finding.find_shortest_path_directions(distance_map, from_loc=self.loc,
                                                                to_locs=home_plant.get_factory_tiles())

        if len(path) > Robot.chain_maxlen:
            log.debug(f'{self.id} the supply path too long: {path}, cannot build a chain')
            self.chain_path = None
        else:
            log.debug(f'{self.id} the supply path has been established: {path}')
            self.chain_path = path
            assert self.chain_path
            # Throw out any SOLAR robots from the path
            for uid, r in robots.items():
                if r.is_mission(Mission.SOLAR) and Robot.dst_locs.get(uid, None) in self.chain_path:
                    log.debug(f'{self.id} throwing out {uid} from the chain path')
                    Robot.dst_locs.pop(uid, None)

            # todo assign support for a supply chain... has to repeat!
            ms = {Mission.RECHARGE: 8, Mission.SOLAR: 6, Mission.GRAZING: 3, Mission.GUARD: 4}
            scores = []
            for uid, r in robots.items():
                if r.home_factory_id == self.home_factory_id:
                    scores.append((uid, ms.get(r.get_mission(), -10) - min(utils.manhattan_distances(r.loc, self.chain_path))))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            if len(scores) >= len(self.chain_path) and scores[len(self.chain_path)-1][1] > 0:
                log.debug(f'{self.id} we have enough robots for the support, assign')
                for uid, s in scores[:len(self.chain_path)]:
                    robots[uid].set_mission(Mission.SUPPORT)

    def unassigned_links(self, robots):
        unassigned = copy.copy(self.chain_path)
        for qid, q in robots.items():
            if q.is_mission(Mission.SUPPORT) and Robot.dst_locs.get(qid, None) in unassigned:
                unassigned.remove(Robot.dst_locs.get(qid, None))
        return unassigned

    def light_links(self, robots):
        if not self.chain_path:
            return []
        assigned = []
        for qid, q in robots.items():
            if q.is_mission(Mission.SUPPORT) and q.is_light() and q.loc in self.chain_path and \
                    Robot.dst_locs.get(qid, None) == q.loc:
                assigned.append(q.loc)
        return assigned

    def missing_links(self, robots):
        if not self.chain_path:
            return []
        missing = copy.copy(self.chain_path)
        for qid, q in robots.items():
            if q.is_mission(Mission.SUPPORT) and q.loc in missing and Robot.dst_locs.get(qid, None) == q.loc:
                missing.remove(q.loc)
        return missing

    def is_chain_locked(self, robots):
        return self.chain_path and not self.missing_links(robots) and \
            utils.is_adjacent(self.loc, self.chain_path[0]) and \
            PlantManager.get_instance().get_plant(self.home_factory_id) is not None

    def get_power_needed_per_step(self):
        return self.power_needed_per_step

    def set_power_needed_per_step(self, p):
        self.power_needed_per_step = p

