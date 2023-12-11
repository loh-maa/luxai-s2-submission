from cachetools import cached, LRUCache
from collections import Counter, defaultdict

import numpy as np

from a09_3a import board_routines
from a09_3a import mission_planner
from a09_3a.mission_allocator import MissionAllocator
from a09_3a.plant import Plant
from a09_3a.plant_manager import PlantManager
from a09_3a.logger import log
from a09_3a.path_finding import NhoodMatrix
from a09_3a.robot import MissionOnTop, Mission, Robot
from a09_3a import utils


class RobotManager:

    def __init__(self, game_state, genome):

        # Constants
        self.player = game_state.player
        self.opp_player = utils.opp_player(game_state.player)
        
        self.genome = genome

        self.robots = {}
        self.mission_planners = {}
        self.mission_allocator = MissionAllocator()

        # Home factory robot counts, for light and heavy (use the weight as the key)
        self.home_factory_robots = {
            1: {},  # {home_plant_id: {robot.id, robot.id, ...}}
            10: {}
        }

        n = game_state.env_cfg.map_size
        self.nhood = NhoodMatrix.get_instance((n, n))

        # Reset static fields in Robot
        Robot.reset()

        # Mission efficiency stats... is a defaultdict(defaultdict(int))
        self.mission_efficiency_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.units_prev = {}
        self.game_state_prev = None

    def update_robots(self, game_state):
        """ I.e. remove destroyed units from variables. Perhaps generalize to update the state of local variables
        at the beginning of each step. """

        plant_man = PlantManager.get_instance()
        units = game_state.units[self.player]

        for uid, unit in units.items():
            if uid not in self.robots:
                produced_factory_id = next((f.unit_id for f in game_state.factories[self.player].values() if f.loc == unit.loc), None)

                # In a rare case, the plant could be destroyed on the same turn it produced the robot
                if produced_factory_id is None:
                    # Use any robot as a temporary substitute
                    robot_tmp = Robot(unit, 'unk', self.genome)
                    new_home_factory_id, _ = PlantManager.get_instance().find_new_home(game_state, robot_tmp, self.home_factory_robots)
                    produced_factory_id = new_home_factory_id

                # If the factory that has produced the robot has less than 10 robots, keep it there, otherwise
                # consider relocation.. remember the factory could have been destroyed on the very same turn it
                # produced the robot...
                home_plant = plant_man.get_plant(produced_factory_id)

                # That's a new unit, create the robot out of it first
                r = Robot(unit, home_plant.id, self.genome)
                self.robots[uid] = r
                self.home_factory_robots[r.weight].setdefault(home_plant.id, set()).add(uid)

            else:
                self.robots[uid].update_unit(unit)

        # Retire destroyed robots and update home_factory_counts
        for uid, robot in list(self.robots.items()):
            if uid not in units:
                robot.retire(game_state)
                self.robots.pop(uid, None)
                self.home_factory_robots[robot.weight][robot.home_factory_id].remove(robot.id)

    def get_l2h_ratio(self, fid):
        n_l = len(self.home_factory_robots[1].get(fid, []))
        n_h = len(self.home_factory_robots[10].get(fid, []))
        return n_l / max(1, n_h)

    def transfer_robot(self, game_state, robot: Robot):

        new_home_factory_id, score = PlantManager.get_instance().find_new_home(game_state, robot, self.home_factory_robots)

        # Check if the route is safe, given the robot's power
        dst_plant = PlantManager.get_instance().get_plant(new_home_factory_id)
        _, cost, path = robot.find_way(game_state, destination_locs=dst_plant.get_factory_tiles())
        # cost = utils.modify_cost_by_solar_gain(game_state, game_state.real_env_steps, cost, robot.weight)
        if robot.power < sum(cost):
            # Cannot transfer safely.. abort
            log.debug(f'@{game_state.real_env_steps} {robot.id} wanted transfer from {robot.home_factory_id} to '
                  f'{new_home_factory_id} (score: {score}), but it has not enough power for safe passage..')
            robot.score_tab.set(Mission.TRANSFER, -50, 50)
            robot.mission_accomplished(Mission.TRANSFER)
            return

        log.debug(f'@{game_state.real_env_steps} transfering {robot.id} from {robot.home_factory_id} to '
                  f'{new_home_factory_id} (score: {score})')
        # Adjust the home factory counts
        self.home_factory_robots[robot.weight][robot.home_factory_id].discard(robot.id)
        # self.home_factory_robots[robot.weight][new_home_factory_id].add(robot.id)
        self.home_factory_robots[robot.weight].setdefault(new_home_factory_id, set()).add(robot.id)
        # Change the home factory in the robot
        robot.home_factory_id = new_home_factory_id
        robot.score_tab.set(Mission.TRANSFER, -100.0, 200)
        robot.mission_accomplished(Mission.TRANSFER)

    @staticmethod
    def calculate_next_locations(game_state, units, new_actions=None):
        """ Can be also used to calculate opponent units planned/intended locations. """
        unit_nlocs = {}
        new_actions = new_actions or {}
        for uid, unit in units.items():
            nloc = Robot.calculate_next_location(unit, game_state, new_actions.get(uid, None))
            unit_nlocs[uid] = nloc
        return unit_nlocs

    @staticmethod
    def update_next_locations(game_state, our_units, new_actions=None):
        """ Update units' next locations based on previously planned actions, do it before planning new moves. """
        next_locs = RobotManager.calculate_next_locations(game_state, our_units, new_actions)
        Robot.next_locs.update(next_locs)

    @cached(cache=LRUCache(maxsize=16), key=lambda self, game_state, weight, home_factory_id: (game_state.ep_step_hash, weight, home_factory_id))
    def n_robots_guarding(self, game_state, weight: int, home_factory_id: str) -> int:
        n = 0
        for uid, robot in self.robots.items():
            if robot.weight == weight and robot.home_factory_id == home_factory_id and robot.is_mission(Mission.GUARD):
                n += 1 if weight == 1 else 3
        return n

    @cached(cache=LRUCache(maxsize=1), key=lambda self, game_state: game_state.ep_step_hash)
    def robots_weight_total(self, game_state):
        return sum([r.weight for r in self.robots.values()])

    def validate_home_plant(self, game_state, robot: Robot):
        pm = PlantManager.get_instance()

        # Make sure we know which factory is the home factory for this robot
        assert robot.home_factory_id
        home_plant = pm.get_plant(robot.home_factory_id)
        if home_plant is None:
            log.debug(f'home factory of {robot.id} has been destroyed')

            new_home_factory_id, score = pm.find_new_home(game_state, robot, self.home_factory_robots)
            assert new_home_factory_id and new_home_factory_id != robot.home_factory_id
            log.debug(f'.. new home factory assigned: {new_home_factory_id}')
            # We also have to change the home factory in the tab
            self.home_factory_robots[robot.weight][robot.home_factory_id].remove(robot.id)
            self.home_factory_robots[robot.weight].setdefault(new_home_factory_id, set()).add(robot.id)

            robot.home_factory_id = new_home_factory_id
            # todo how to clear actions?
            if robot.is_mission_any([Mission.SOLAR, Mission.SUPPORT, Mission.GRAZING, Mission.GUARD]):
                robot.set_mission(Mission.NONE)
            home_plant = pm.get_plant(new_home_factory_id)

        assert isinstance(home_plant, Plant)
        assert home_plant.id == robot.home_factory_id

        return home_plant

    def validate_heavy_roles(self, game_state):
        """ Wow this is so tricky... any simpler solution to keep the role assignment valid? """

        for fid, f in PlantManager.get_instance().our_plants.items():
            heavies = [r for r in self.robots.values() if r.is_heavy() and r.home_factory_id == fid]
            primary_ice = [r for r in heavies if r.is_primary_ice]
            primary_ore = [r for r in heavies if r.is_primary_ore]

            if len(heavies) == 1:
                heavies[0].is_primary_ice = True
                heavies[0].is_primary_ore = True

            elif len(heavies) > 1:

                if not primary_ice:
                    # Primary ice robot is missing, assign to any heavy that is not primary ore yet
                    for r in heavies:
                        if not r.is_primary_ore:
                            r.is_primary_ice = True
                            break
                elif len(primary_ice) > 1:
                    for r in primary_ice[1:]:
                        r.is_primary_ice = False

                if not primary_ore:
                    for r in heavies:
                        if not r.is_primary_ice:
                            r.is_primary_ore = True
                            break
                elif len(primary_ore) > 1:
                    for r in primary_ore[1:]:
                        r.is_primary_ore = False

                # At this point we are guaranteed that 1 and only 1 robot is assigned ice and 1 and only 1 is
                # assigned ore, but... they may be the same robot!

                primary_ice = [r for r in heavies if r.is_primary_ice]
                primary_ore = [r for r in heavies if r.is_primary_ore]

                if primary_ice[0] == primary_ore[0]:
                    # Split the role, first check if any of the robots is on a ICE or ORE mission, assign accordingly..
                    assert len(primary_ice) == len(primary_ore) == 1
                    new_robot = [r for r in heavies if r != primary_ice[0]][0]
                    if primary_ice[0].is_mission(Mission.GO_ORE):
                        primary_ice[0].is_primary_ice = False
                        new_robot.is_primary_ice = True
                    else:
                        primary_ice[0].is_primary_ore = False
                        new_robot.is_primary_ore = True

        roles = [(r.id, r.home_factory_id, r.is_primary_ice, r.is_primary_ore)
                 for r in self.robots.values() if r.is_heavy()]
        log.debug(f'@{game_state.real_env_steps} heavy roles: {roles}')

    def check_assign_emergency_water_mission(self, game_state):
        # Factory call for a mission
        for fid, plant in PlantManager.get_instance().our_plants.items():
            if plant.needs_emergency_water() and fid not in Robot.emergency_water_robots.values():
                log.debug(f'@{game_state.real_env_steps} {fid} needs emergency water...')
                # Find the best delivery
                costs = {}
                for uid, robot in self.robots.items():
                    if robot.is_light():
                        costs[uid] = robot.estimate_water_delivery_cost(plant)
                    else:
                        # In a rare case we have only heavy robots left, make sure costs is not empty
                        costs[uid] = 1000
                # Select the robot and assign it the special mission
                sel_uid = min(costs, key=(lambda key: costs[key]))
                if costs[sel_uid] < 1000:
                    log.debug(f'\t... {sel_uid} assigned')
                    # self.robots[sel_uid].set_mission(Mission.BRING_WATER, dst_plant_id=plant.id)
                    self.robots[sel_uid].set_mission(Mission.BRING_WATER)
                    robot = self.robots[sel_uid]
                    home_plant = PlantManager.get_instance().get_plant(robot.home_factory_id)
                    self.mission_planners[sel_uid] = mission_planner.MissionPlannerBringWater(game_state, robot, home_plant, dst_plant_id=plant.id)

                    Robot.emergency_water_robots[sel_uid] = fid
                else:
                    log.debug(f'\t... no suitable robot found.. scores: {costs}')

    def check_assign_hunt_mission(self, game_state):

        # Iterate over all enemy units, find units that are "lost" and try to assign our robot to hunt them down.
        # Also find opponent units that threaten/sabotage our lichen and assign our robots to engage them.

        ofdm = board_routines.get_factories_distance_map(game_state, game_state.opp_player)
        olua = PlantManager.get_instance().get_our_lichen_under_attack_map()
        pinm = PlantManager.get_instance().get_primary_ice_neighborhood_map(game_state)
        tadv = board_routines.get_territorial_advantage(game_state)
        step = game_state.real_env_steps

        # Our map of aggressive units
        our_agg_robots = [r for r in self.robots.values() if r.is_aggressive() and r.id not in Robot.preys_targeted]
        if not our_agg_robots:
            # We don't have any aggressive robots at hand
            return

        for oid, onit in game_state.units[game_state.opp_player].items():
            vulnerability = ofdm[onit.loc] / 100 - onit.power / onit.unit_cfg.BATTERY_CAPACITY

            # If the unit is vulnerable or if it attacks our lichen
            if vulnerability > 0.0 or olua[onit.loc] or pinm[onit.loc]:

                # How urgent it is to engage the unit?

                already_targeted = len([rid for rid, pid in Robot.preys_targeted.items() if pid == oid])

                # Try to find a good match with our unit
                scores = {}
                for r in our_agg_robots:
                    if onit.weight == r.weight and 0.33 < r.power2b():
                        l_dist = utils.manhattan_distance(r.loc, onit.loc)
                        if l_dist > 12:
                            continue
                        # OU relative distance from OF to the distance from our robot, it only starts to get interesting
                        # when it's >4
                        score = ofdm[onit.loc] / l_dist
                        score += 7 - l_dist
                        score += 30 * min(0.2, (r.power - onit.power)/r.battery)
                        score += 5 * olua[onit.loc]
                        score -= 7 * already_targeted
                        # Decrease chasing after opp units if it's not strictly defense, in proximity of our factory
                        score += 3 * min(0.0, tadv[onit.loc]) * utils.hi_pass(step, 900, 950)
                        scores[r.id] = score

                if scores:
                    best_id = max(scores.keys(), key=lambda k: scores[k])
                    log.debug(f'best match to engage opponent\'s {oid} is {best_id}: {scores[best_id]}')

                    if scores[best_id] > 9:
                        log.debug('\t... engaging')
                        robot = self.robots[best_id]
                        robot.set_mission(Mission.HUNT)
                        self.mission_planners[robot.id] = mission_planner.MissionPlannerHunt(game_state, robot, prey_id=oid)

                log.debug(f'vulnerable or trespassing opponent {oid} currently targeted by: {already_targeted}')

    def assign_primary_missions(self, game_state):
        for fid, plant in PlantManager.get_instance().our_plants.items():
            if 120 < plant.cargo.water:
                # Only for factories that are separated from opponent's plants
                flocs = board_routines.get_factory_locs(game_state, game_state.player) + \
                        board_routines.get_factory_locs(game_state, game_state.opp_player)
                pil = plant.primary_ice_loc or plant.loc
                n_opp_factories_nearby = sum([utils.manhattan_distance(pil, floc) < 10 for floc in flocs])
                if plant.get_ice_diversity() < 0.13 * n_opp_factories_nearby:
                    log.debug(f'{fid} ice diversity: {plant.get_ice_diversity()}, n opp factories nearby: '
                              f'{n_opp_factories_nearby}, skip ore run')
                    continue

                for uid, r in self.robots.items():
                    if r.home_factory_id == fid and r.is_primary_ice and r.is_primary_ore and r.is_mission_any([Mission.NONE, Mission.GO_ICE]):
                        # Choose between ice and ore
                        score, dst_loc, steps_to_wait = r.evaluate_initial_go_ore(game_state, plant)
                        if score > -20.0:
                            r.set_mission(Mission.WAIT)
                            self.mission_planners[r.id] = mission_planner.MissionPlannerWait(game_state, r, plant, steps_to_wait, Mission.GO_ORE)

    def assign_light_sabotage(self, game_state):
        # Assign only once! Otherwise we may spiral into expensive loops
        if game_state.real_env_steps == 950:
            for uid, r in self.robots.items():
                if r.is_light():
                    r.set_mission(Mission.SABOTAGE)

    def calculate_mission_scores(self, game_state, robot, home_plant):

        step = game_state.real_env_steps
        mission_scores = {}

        mission_scores[Mission.GO_HOME] = self.genome.mission_weights.get('home', 1.0) * (
            25 * utils.low_pass(robot.power2b(), 0.05, 0.4) *
                utils.hi_pass(home_plant.power, 3000, 6000) *
                utils.low_pass(step, 850, 930) *
                utils.low_pass(utils.manhattan_distance(robot.loc, home_plant.loc), 10, 20) +
            7 * utils.hi_pass((robot.cargo.ice + robot.cargo.ore)/robot.weight, 30, 80) +
            -10 * int(robot.is_light())
        )

        _, transfer_score = PlantManager.get_instance().find_new_home(game_state, robot, self.home_factory_robots)
        mission_scores[Mission.TRANSFER] = self.genome.mission_weights.get('transfer', 1.0) * (
            30 * utils.hi_pass(transfer_score, 3, 20) +
            -30 * utils.hi_pass(step, 800, 900) +
            -30 * (robot.is_primary_ice or robot.is_primary_ore) +
            robot.score_tab.get(Mission.TRANSFER)
        )

        primary_ice_source_free = home_plant.primary_ice_loc is not None and home_plant.primary_ice_loc not in Robot.dst_locs.values()
        ice_score, _ = robot.evaluate_mission_go_res(game_state, home_plant, resource='ice')
        ice_demand = max(utils.low_pass(home_plant.cargo.water, 75, 200) * utils.low_pass(step, 850, 950),
                         utils.low_pass(home_plant.water_balance, -50, 100) * utils.low_pass(step, 870, 930))
        ice_demand = max(ice_demand, utils.low_pass(step, 0, 800)**2)
        mission_scores[Mission.GO_ICE] = self.genome.mission_weights.get('ice', 1.0) * (
            60 * int(robot.is_primary_ice and primary_ice_source_free) * ice_demand +
            50 * int(robot.is_primary_ice) * utils.hi_pass(ice_score, 0.3, 1.0) * ice_demand +
            -40 * utils.low_pass(ice_score, -0.7, 0.0) +
            25 * utils.low_pass(home_plant.cargo.water, 100, 200) * utils.low_pass(step, 800, 900) +
            10 * utils.low_pass(home_plant.cargo.water, 0, 1000 - step) +
            -10 * utils.low_pass(step, 50, 100) * int(robot.is_light()) +
            -30 * utils.low_pass(robot.power2b(), 0.2, 0.5) +
            robot.score_tab.get(Mission.GO_ICE)
        )

        # Reminder the ice_score and ore_score take robot's power into account,
        # at 20% power the score is low even for a perfect digging tile
        ore_score, _ = robot.evaluate_mission_go_res(game_state, home_plant, resource='ore')
        mission_scores[Mission.GO_ORE] = self.genome.mission_weights.get('ore', 1.0) * (
            50 * utils.hi_pass(ore_score, 0.3, 1.0) * int(robot.is_primary_ore) +
            30 * utils.hi_pass(ore_score, 0.3, 1.0) * int(robot.is_primary_ore) * utils.low_pass(step, 100, 200) +
            -40 * utils.low_pass(ore_score, -0.6, 0.4) +
            40 * utils.hi_pass(home_plant.cargo.water, 100, 200) * int(robot.is_primary_ore) * utils.low_pass(step, 150, 250) +
            -100 * utils.hi_pass(step, 650, 750) +
            -15 * utils.hi_pass(home_plant.cargo.metal, 150, 250) +
            20 * utils.hi_pass(home_plant.power, 5000, 10000) * utils.low_pass(step, 500, 700) +
            -20 * utils.low_pass(robot.power2b(), 0.2, 0.5) +
            robot.score_tab.get(Mission.GO_ORE)
        )

        n_lichen_tiles = home_plant.n_lichen_tiles(game_state)
        hf_under_attack = home_plant.under_attack(game_state)
        mission_scores[Mission.GUARD] = self.genome.mission_weights.get('guard', 1.0) * (
            15 * utils.hi_pass(robot.power2b(), 0.15, 0.3) * utils.low_pass(robot.power2b(), 0.5, 0.9) +
            15 * utils.low_pass((1 + self.n_robots_guarding(game_state, robot.weight, home_plant.id)) / (1 + n_lichen_tiles), 0.1, 0.5) +
            -10 * utils.hi_pass(home_plant.power, 4000, 8000) * utils.low_pass(step, 650, 850) +
            15 * hf_under_attack * utils.hi_pass(step, 750, 950) +
            -45 * int(robot.is_primary_ice or robot.is_primary_ore) +
            robot.score_tab.get(Mission.GUARD)
        )

        fct = home_plant.get_connection_tiles()
        n_clear_connection_tiles = np.sum(game_state.board.rubble[tuple(fct.T)] == 0)
        grazing_demand = utils.low_pass(home_plant.dominance_area, 20, 100) * utils.hi_pass(home_plant.water_balance, 50, 300)
        mission_scores[Mission.GRAZING] = self.genome.mission_weights.get('grazing', 1.0) * (
            5 * min(utils.low_pass(step, 940, 980), utils.hi_pass(step, 200, 500)) +
            5 * min(utils.low_pass(step, 940, 980), utils.hi_pass(step, 500, 700)) +
            20 * grazing_demand * (1.5 if robot.is_light() else 0.8) +
            15 * utils.low_pass(n_clear_connection_tiles, 1, 9) +
            -50 * int(robot.is_primary_ice or robot.is_primary_ore) * utils.low_pass(step, 700, 900) +
            robot.score_tab.get(Mission.GRAZING)
        )

        mission_scores[Mission.SABOTAGE] = self.genome.mission_weights.get('sabotage', 2.0) * (
            20 * utils.hi_pass(step, 200, 300) +
            20 * int(robot.is_heavy()) * utils.hi_pass(len(self.home_factory_robots[robot.weight].get(robot.home_factory_id, [])), 2, 5) +
            20 * int(robot.is_light()) * utils.hi_pass(len(self.home_factory_robots[robot.weight].get(robot.home_factory_id, [])), 7, 14) * int(step > 800) +
            -20 * int(robot.power2b() < 0.5 and (not home_plant.is_factory_tile(robot.loc) or home_plant.power < 3000)) * utils.low_pass(step, 820, 920) +
            -60 * int(robot.is_primary_ice or robot.is_primary_ore) +
            robot.score_tab.get(Mission.SABOTAGE)
        )

        mission_scores[Mission.SOLAR] = self.genome.mission_weights.get('solar', 1.0) * (
            -40 * int(robot.is_primary_ice or robot.is_primary_ore or robot.is_light()) +
            -40 * int(step > 600) +
            20 * utils.low_pass(step, 400, 600) * int(robot.is_heavy()) +
            10 * utils.low_pass(robot.power2b(), 0.1, 0.3) +
            10 * utils.hi_pass(len(self.home_factory_robots[10].get(robot.home_factory_id, {})), 3, 5) +
            robot.score_tab.get(Mission.SOLAR)
        )

        stay_at_factory = robot.power < robot.battery // 10 and robot.loc in board_routines.get_our_factory_tiles(game_state)
        mission_scores[Mission.RECHARGE] = self.genome.mission_weights.get('recharge', 1.0) * (
            -45 * int(robot.is_primary_ice or robot.is_primary_ore) +
            30 * int(stay_at_factory) +
            10 * utils.low_pass(robot.power2b(), 0.1, 0.3) * utils.low_pass(step, 800, 950) +
            5 * utils.low_pass(robot.power2b(), 0.3, 0.7) * utils.low_pass(step, 750, 900)
        )

        mission_scores[Mission.SUPPORT] = self.genome.mission_weights.get('support', 1.0) * (
            -40 * int(robot.is_primary_ice or robot.is_primary_ore) +
            40 +
            robot.score_tab.get(Mission.SUPPORT)
        )

        return mission_scores

    def determine_actions(self, game_state, robot, home_plant):
        step = game_state.real_env_steps
        actions = None
        unit = robot.unit
        plant_man = PlantManager.get_instance()

        assert robot.home_factory_id == home_plant.id
        log.debug(f'@{step} {robot} ({home_plant.id}): {robot.get_mission()}')

        # A few mission-independent singular actions
        assert home_plant is not None

        # Find out if the robot is at any of our plants
        at_plant = next((p for p in plant_man.our_plants.values() if p.is_factory_tile(unit.loc)), None)

        # 1. Get extra power from factory if necessary and possible
        # if at_plant and not robot.power_pickup_in_queue() and not robot.cargo_transfer_in_queue() and \
        #         not robot.is_mission_any([Mission.SUPPORT, Mission.SOLAR]):
        if at_plant and not robot.is_mission_any([Mission.SUPPORT, Mission.SOLAR]):
            # Robot is at our plant, power up if:
            # The plant has plenty of power
            # The unit is the main ice supplier (or is going to the main ice source)
            # The light robot is very low on power and the factory becomes congested
            plant_has_plenty_of_power = at_plant.has_plenty_of_power(game_state, robot)
            robot_has_mission_critical = robot.has_mission_critical()
            is_primary = robot.is_primary_ice or robot.is_primary_ore

            if (robot_has_mission_critical or is_primary and robot.is_idle() or robot.is_mission(Mission.SABOTAGE)) and robot.power2b() < 0.8:
                power_to_pick = min(at_plant.power, robot.battery - robot.power)

            elif plant_has_plenty_of_power and robot.power2b() < 0.67 and not robot.score_tab.get(Mission.POWER_PICKUP):
                power_to_pick = min(0.33 * at_plant.power, robot.battery - robot.power)
                limit = (1000 - step) * 1.5 * robot.avg_sabotage_cost if step > 850 else robot.battery
                power_to_pick = min(power_to_pick, limit)

            else:
                power_to_pick = 0

            if power_to_pick > 0.1 * robot.battery:
                log.debug(f'{robot.id} at {at_plant.id} taking up power (condition 1)..')
                actions = [unit.pickup(pickup_resource=4, pickup_amount=int(power_to_pick))]
                robot.score_tab.set(Mission.POWER_PICKUP, score=-1.0, turns=20)
                return actions

            robot_low_on_power = robot.power2b() < 0.1
            plant_has_enough = robot.battery // 10 < at_plant.power
            # Charge the heavy robot and send it away
            if robot_low_on_power and plant_has_enough and not (robot.is_primary_ice or robot.is_primary_ore):
                # At the end, don't take more than reasonably can spend
                power_to_pick = robot.battery // 10
                log.debug(f'{robot.id} at {at_plant.id} taking up power (condition 2)..')
                actions = [unit.pickup(pickup_resource=4, pickup_amount=power_to_pick)]
                return actions

        # 2. If has some ice or ore and inside factory, dump the cargo
        if unit.cargo.ice > 0 and at_plant and not robot.cargo_transfer_in_queue():
            if unit.power >= robot.action_queue_cost:
                actions = [unit.transfer(0, transfer_resource=utils.ResourceId.ICE.value, transfer_amount=unit.cargo.ice)]

        elif unit.cargo.ore > 0 and at_plant and not robot.cargo_transfer_in_queue():
            if unit.power >= robot.action_queue_cost:
                actions = [unit.transfer(0, transfer_resource=utils.ResourceId.ORE.value, transfer_amount=unit.cargo.ore)]

        # Unload water automatically, but not if actually on the BRING_WATER mission
        elif unit.cargo.water > 0 and at_plant and not robot.is_mission(Mission.BRING_WATER):
            if unit.power >= robot.action_queue_cost:
                actions = [unit.transfer(0, transfer_resource=utils.ResourceId.WATER.value, transfer_amount=unit.cargo.water)]

        elif robot.is_mission(Mission.BRING_WATER):
            mp = self.mission_planners.get(robot.id, None)
            if not isinstance(mp, mission_planner.MissionPlannerBringWater):
                log.error('BRING_WATER mission probably interrupted by another mission and the MissionPlanner has been '
                          'overwritten.. do not interrupt BRING_WATER missions!')
                assert False
            actions = mp.plan(game_state)

        elif robot.is_mission(Mission.ATTACK):
            mp = self.mission_planners.get(robot.id, None)
            assert isinstance(mp, mission_planner.MissionPlannerAttack)
            actions = self.mission_planners[robot.id].plan(game_state)

        elif robot.is_mission(Mission.GO_HOME):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerGoHome):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerGoHome(game_state, robot, home_plant)
            actions = self.mission_planners[robot.id].plan(game_state)

        elif robot.is_mission(Mission.GO_ICE):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerGoIce):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerGoIce(game_state, robot, home_plant)
            self.mission_planners[robot.id].update_robots(self.robots)
            actions = self.mission_planners[robot.id].plan(game_state)
            if actions is False:
                # Mission accomplished, but we need to reset the action loop
                actions = []

        elif robot.is_mission(Mission.GO_ORE):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerGoOre):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerGoOre(game_state, robot, home_plant)
            self.mission_planners[robot.id].update_robots(self.robots)
            actions = self.mission_planners[robot.id].plan(game_state)
            if actions is False:
                # Mission accomplished, but we need to reset the action loop
                actions = []

        elif robot.is_mission(Mission.GUARD):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerGuard):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerGuard(game_state, robot, home_plant)
            mp = self.mission_planners[robot.id]
            actions = mp.plan(game_state)

        elif robot.is_mission(Mission.GRAZING):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerGraze):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerGraze(game_state, robot, home_plant)
            actions = self.mission_planners[robot.id].plan(game_state)

        elif robot.is_mission(Mission.HUNT):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerHunt):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerHunt(game_state, robot, home_plant)
            actions = self.mission_planners[robot.id].plan(game_state)

        elif robot.is_mission(Mission.KAMIKAZE):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerKamikaze):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerKamikaze(game_state, robot, home_plant)
            actions = self.mission_planners[robot.id].plan(game_state)

        elif robot.is_mission(Mission.RECHARGE):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerRecharge):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerRecharge(game_state, robot, home_plant)
            actions = self.mission_planners[robot.id].plan(game_state)

        elif robot.is_mission(Mission.SABOTAGE):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerSabotage):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerSabotage(game_state, robot, home_plant)
            actions = self.mission_planners[robot.id].plan(game_state)

        elif robot.is_mission(Mission.SOLAR):

            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerSolar):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerSolar(game_state, robot, home_plant)
            actions = self.mission_planners[robot.id].plan(game_state)
            if actions is False:
                # Mission accomplished, but we need to reset the action loop
                actions = []

        elif robot.is_mission(Mission.SUPPORT):
            mp = self.mission_planners.get(robot.id, None)
            if mp is None or not isinstance(mp, mission_planner.MissionPlannerSupport):
                self.mission_planners[robot.id] = mission_planner.MissionPlannerSupport(game_state, robot, home_plant)
            self.mission_planners[robot.id].update_robots(self.robots)
            actions = self.mission_planners[robot.id].plan(game_state)
            if actions is False:
                # Mission accomplished, but we need to reset the action loop
                actions = []

        elif robot.is_mission(Mission.TRANSFER):
            # Just administration, no actions required
            self.transfer_robot(game_state, robot)
            actions = False

        elif robot.is_mission(Mission.WAIT):
            mp = self.mission_planners.get(robot.id, None)
            if not isinstance(mp, mission_planner.MissionPlannerWait):
                return False
            actions = mp.plan(game_state)

        return actions

    def plan_missions(self, game_state, remainingOverageTime):
        units = game_state.units[self.player]
        step = game_state.real_env_steps
        actions = dict()

        # First update all robots and their expected locations
        self.update_robots(game_state)
        self.update_next_locations(game_state, units)

        if step < 850:
            # We don't care about the roles in the end game
            self.validate_heavy_roles(game_state)
        self.check_assign_emergency_water_mission(game_state)
        self.check_assign_hunt_mission(game_state)
        if step < 120:
            self.assign_primary_missions(game_state)

        if 850 <= step <= 970 and step % 10 == 0:
            self.mission_allocator.strategic_matrix(game_state, self.robots)
            mp = self.mission_allocator.assign_attack(game_state, self.robots)
            self.mission_planners.update(mp)

        for uid, robot in self.robots.items():

            mission_scores = None
            for attempt in range(2):

                # Make sure we know which factory is the home factory for this robot, we need to get the home plant
                # at every iteration, because it could change
                home_plant = self.validate_home_plant(game_state, robot)

                try:
                    # todo
                    if robot.is_mission(Mission.NONE) or \
                        (robot.is_mission_any([Mission.GRAZING, Mission.GUARD, Mission.RECHARGE]) and np.random.random() < 0.1):

                        # Assign a score for each mission type given the context,
                        # and select mission with the highest score
                        mission_scores = self.calculate_mission_scores(game_state, robot, home_plant)

                        log.debug(f'@{step} {uid} select a new mission, scores: {mission_scores}')
                        # assert Mission.SABOTAGE in mission_scores

                        if self.genome.flags['probabilistic_mission']:
                            log.debug('Select mission probabilistically..')
                            missions, scores = zip(*mission_scores.items())
                            scores = np.clip(np.array(scores), 0, 999) ** self.genome.params['probabilistic_mission_exponent']
                            if np.sum(scores) > 0.01:
                                new_mission = np.random.choice(missions, p=scores / np.sum(scores))
                            else:
                                log.debug('no good candidate mission found')
                                break
                        else:
                            log.debug('Select mission deterministically..')
                            mission = max(mission_scores.keys(), key=lambda k: mission_scores[k])
                            if mission_scores[mission] > 0:
                                new_mission = mission
                            else:
                                log.debug('no good candidate mission found')
                                break

                        if new_mission != robot.get_mission():
                            robot.set_mission(new_mission)
                            self.mission_planners[robot.id] = None

                    actions[uid] = self.determine_actions(game_state, robot, home_plant)

                    if actions[uid] is False:
                        # Actions unresolved, try again...
                        del actions[uid]
                    else:
                        # Actions resolved
                        break

                except MissionOnTop as e:
                    # if attempt >= 4:
                    #     log.warning(f'@{step} {uid}: action resolution failed')
                    #     break
                    # Otherwise, just try again to resolve
                    robot.set_mission(e.new_mission)
                    if e.new_mission == Mission.RECHARGE:
                        self.mission_planners[robot.id] = mission_planner.MissionPlannerRecharge(game_state, robot, home_plant, recharge_to=e.kwargs['recharge_to'])

        return actions

    def check_for_friendly_collisions(self):
        """ Check for collisions based on the updated next location anticipation. """

        assert set(self.robots.keys()) == set(Robot.next_locs.keys())

        # Create a reverse dictionary with a list of robots coming to the same loc
        nloc_uids = defaultdict(list)
        for uid, nloc in Robot.next_locs.items():
            nloc_uids[nloc].append(uid)

        modify_actions_for = []
        for nloc, uids in nloc_uids.items():
            if len(uids) > 1:
                log.debug(f'friendly collision expected at {nloc} for {uids}')
                # How to resolve..?
                # 1. When equal or lighter robot is moving, cancel its action
                # 2. When two or more equal robots are moving, cancel all but one of them
                # 3. When heavy robot wants to go on a light robot tile, move the lighter robot away
                ump = []
                for uid in uids:
                    r = self.robots[uid]
                    is_moving = r.loc != nloc
                    priority = 3 * int(r.is_heavy()) - int(is_moving) - r.power2b() + \
                               (r.unit.cargo.ice + r.unit.cargo.ore) / r.unit.unit_cfg.CARGO_SPACE + \
                               7 * int(r.has_mission_critical()) + \
                               int(r.is_mission_any([Mission.GO_HOME, Mission.GO_ICE, Mission.GO_ORE])) + \
                               5 * int(r.is_mission([Mission.SUPPORT])) - int(r.is_mission([Mission.SOLAR]))

                    ump.append((uid, is_moving, priority))

                ump = sorted(ump, key=lambda x: x[2])
                # Modify actions for all except the highest priority
                modify_actions_for += ump[:-1]

        return modify_actions_for

    def collision_avoidance_with_our_units(self, game_state, actions):
        actions_mod = {}
        risk_map = board_routines.get_risk_map(game_state)

        # Check for our own collisions, overwrite the already queued actions with the newly planned actions
        # Also, to avoid cycles in collision resolution, save the previously planned next location as a NO-GO
        modify_actions_for = self.check_for_friendly_collisions()
        if modify_actions_for:
            log.debug(f'modify actions for: {modify_actions_for}')
            for (uid, is_moving, _) in modify_actions_for:
                if is_moving:
                    if actions.get(uid, None) is not None:
                        # If the action is ONLY newly planned, set to None
                        actions_mod[uid] = None
                    else:
                        # If the action is already in the queue, assign a new action
                        actions_mod[uid] = [self.robots[uid].recharge(0)]
                else:
                    # The agent is not moving, it's most probably a light robot standing in a way of a heavy one
                    robot = self.robots[uid]
                    safe_loc = self.find_safe_adjacent(robot, game_state, risk_map)
                    # Free location found, but try to move only if we actually can move, otherwise we end up swaying
                    if safe_loc:
                        can_move = robot.can_move_onto(game_state, safe_loc)
                        if can_move:
                            log.debug(f'{uid} moving to a free loc: {safe_loc}')
                            direction = utils.direction_to(robot.loc, safe_loc)
                            actions_mod[uid] = [robot.unit.move(direction)]
                        else:
                            log.debug(f'{uid} cannot move to a free loc: {safe_loc}.. power: {robot.power}')
                    else:
                        log.debug(f'{uid} no free location found')

        return actions_mod

    def find_safe_adjacent(self, robot, game_state, risk_map):
        """ Free from our units. Safe, means we are not trying to engage the opponent. """
        free_locs = []
        # Prefer tiles with low rubber, our factory tiles or in direction of our closest factory
        oflocs = board_routines.get_factory_locs(game_state, game_state.player)
        d_oflocs = min(utils.manhattan_distances(robot.loc, oflocs))
        our_factory_tiles = board_routines.get_our_factory_tiles(game_state)
        opp_factory_tiles = board_routines.get_opp_factory_tiles(game_state)

        for adj_loc in self.nhood[robot.loc]:
            if adj_loc not in opp_factory_tiles:
                rub = game_state.board.rubble[adj_loc]
                if adj_loc not in Robot.next_locs.values():
                    # Prefer our factory locations
                    if adj_loc in our_factory_tiles:
                        return adj_loc
                    elif risk_map[adj_loc] <= robot.pweight:
                        if robot.is_mission_any([Mission.ATTACK, Mission.KAMIKAZE]) and robot.power2b() < 1.1 and \
                                utils.ab_shorter_than_bc(adj_loc, Robot.dst_locs.get(robot.id, robot.loc), robot.loc):
                            # If attacking (and low on power?), try to get to your destination, instead of getting away
                            priority = 500 - rub + np.random.normal(loc=0.0, scale=10.0)
                        elif robot.power2b() < 0.25 and robot.is_mission(Mission.GO_HOME) and \
                                min(utils.manhattan_distances(adj_loc, oflocs)) < d_oflocs:
                            # If low on power, prefer tiles in direction of our closest factory
                            priority = 700 - 500 * robot.power2b() - rub + np.random.normal(loc=0.0, scale=10.0)
                        elif risk_map[adj_loc] == 5:
                            # If the opp robot is stationary (and we already have higher pweight), step in
                            priority = 100 + 20 * robot.power2b() - rub + np.random.normal(loc=0.0, scale=10.0)
                        else:
                            # Else, just an ordinary field
                            priority = 100 + 10 - rub + np.random.normal(loc=0.0, scale=10.0)
                    else:
                        # Take the risk of stepping on stronger opponent robot
                        priority = 20 - rub / 20 + np.random.normal(loc=0.0, scale=1.0)
                else:
                    # The location candidate is going to be taken by our other robot... so that's a last resort
                    priority = -10 - rub / 100

                free_locs.append((tuple(adj_loc), priority))

        if free_locs:
            loc_sel, _ = max(free_locs, key=lambda x: x[1])
            return loc_sel
        else:
            return None

    def collision_avoidance_with_opponent_units(self, game_state):
        step = game_state.real_env_steps
        steps_left = 999 - step
        actions_mod = {}

        our_factory_tiles = board_routines.get_our_factory_tiles(game_state)
        risk_map = board_routines.get_risk_map(game_state)

        # If bigger or equal robots try to crash into our unit (which is not in a factory) or even if they can
        # Try to move into their location
        assert set(self.robots.keys()) == set(Robot.next_locs.keys())
        for uid, nloc in Robot.next_locs.items():
            robot = self.robots[uid]

            if robot.is_light() and robot.is_mission(Mission.KAMIKAZE):
                if 4.5 < risk_map[nloc] < 5.5:
                    if board_routines.get_lichen_map(game_state, game_state.opp_player)[nloc] > 20 * steps_left:
                        continue
                    elif board_routines.get_lichen_map(game_state, game_state.opp_player)[robot.loc] > 20 * steps_left:
                        log.debug(f'{uid} self-destruct order, loc: {robot.loc}')
                        actions_mod[uid] = [robot.unit.self_destruct(repeat=1, n=1)]
                        continue
                    else:
                        # Just relax and go ahead (==continue) with 50%
                        if np.random.random() < 0.5:
                            continue
                else:
                    continue

            # Do avoidance only if we are at risk of being crushed
            # Specifically, skip this robot if:
            # - it's stepping into our factory
            # - it's power-heavier than the risk
            # - it's equal weight to the risk but is moving and has more power
            if nloc in our_factory_tiles or \
                    risk_map[nloc] < robot.pweight and nloc != robot.loc or \
                    nloc == robot.loc and risk_map[nloc] < robot.weight:
                # We are safe moving on to the next location.. however still, if this is a fight over a resource tile
                # we may do better by stepping in probabilistically
                continue

            # Do avoidance if:
            # - the robot is power-lighter than the risk
            else:
                # Scrap our next location plan, stay put or move somewhere else
                log.debug(f'@{step} {uid} possible collision with opponent, nloc: {nloc}, pweight: {risk_map[nloc]}')
                if risk_map[robot.loc] < robot.weight:
                    # We are safe if we stay put, but sometimes we just need to go through, so balance probabilistically
                    # if robot.n_turns_avoiding_collision > 50 and robot.unit.power > robot.unit.unit_cfg.BATTERY_CAPACITY // 5:
                    #     log.debug(f'.. we have probably stayed too long here, abort the mission, do something else')
                    #     robot.mission_accomplished(robot.get_mission())
                    if robot.weight == 1 and risk_map[nloc] < 5 and robot.is_mission(Mission.BRING_WATER):
                        # A light robot is being threatened by opponent's light unit.. take the risk randomly
                        if np.random.random() < 0.25:
                            log.debug(f'{uid} col avoidance: keep going')
                            # actions_mod[uid] = None
                            continue
                        else:
                            log.debug(f'{uid} col avoidance: stop')
                            actions_mod[uid] = []
                    elif robot.weight == 1 and risk_map[nloc] > 5:
                        if np.random.random() < 0.2:
                            log.debug(f'{uid} col avoidance: keep going')
                            continue
                        else:
                            log.debug(f'{uid} col avoidance: stop')
                            actions_mod[uid] = []
                    elif robot.weight == 1 and risk_map[nloc] > 1:
                        if np.random.random() < 0.15:
                            log.debug(f'{uid} col avoidance: keep going')
                            continue
                        else:
                            log.debug(f'{uid} col avoidance: stop')
                            actions_mod[uid] = []
                    elif robot.weight == 10 and risk_map[nloc] > 9:
                        if np.random.random() < 0.15:
                            log.debug(f'{uid} col avoidance: keep going')
                            continue
                        else:
                            log.debug(f'{uid} col avoidance: stop')
                            actions_mod[uid] = []
                    else:
                        log.debug(f'{uid} stay put/recharge')
                        actions_mod[uid] = []

                else:

                    safe_loc = self.find_safe_adjacent(robot, game_state, risk_map)
                    # Free location found, but try to move only if we actually can move, otherwise we end up swaying
                    if safe_loc:
                        can_move = robot.can_move_onto(game_state, safe_loc)
                        if can_move:
                            log.debug(f'{uid} moving to a safe loc: {safe_loc}')
                            direction = utils.direction_to(robot.loc, safe_loc)
                            actions_mod[uid] = [robot.unit.move(direction)]
                        else:
                            log.debug(f'{uid} cannot move to a free loc: {safe_loc}.. power: {robot.power}')
                    else:
                        log.debug(f'{uid} no free location found')

        return actions_mod

    def ram_opponent_units(self, game_state):
        actions_mod = {}
        opp_units = game_state.units[self.opp_player]
        opp_factory_tiles = board_routines.get_opp_factory_tiles(game_state)
        opp_unit_nlocs = self.calculate_next_locations(game_state, opp_units)
        # If the opponent unit is stationary, try to move into their current location
        # this may work for unprepared opponents, for advanced opponents, we must continue only if:
        # 1. We have more power left than the opponent

        for uid, robot in self.robots.items():
            if not robot.is_aggressive():
                continue

            if robot.power2b() >= 0.25 and np.random.random() < 0.3:

                for oid, onloc in opp_unit_nlocs.items():
                    # Let's our heavy robots not chase the light robots
                    if utils.is_adjacent(robot.loc, onloc) and \
                            (robot.weight == opp_units[oid].weight or
                             robot.weight > opp_units[oid].weight and np.random.random() < 0.33) and \
                            robot.pweight > opp_units[oid].get_power_weight() and \
                            onloc not in opp_factory_tiles:
                        log.debug(f'Our {uid} wants to ram {oid}.. but can it move into {onloc}?...')
                        if robot.can_move_onto(game_state, onloc):
                            log.debug(f'yes, {uid} trying to ram into {onloc} ...')
                            direction = utils.direction_to(robot.loc, onloc)
                            actions_mod[uid] = [robot.unit.move(direction)]
                        else:
                            log.debug(f'... no')

        return actions_mod

    def avoid_guarded_res_tiles(self, game_state):
        """ If an agent wants to dig a resource or sabotage enemy lichen, but the tile is 'guarded', just stay aside.
        """
        actions_mod = {}
        risk_map = board_routines.get_risk_map(game_state)
        opp_lichen = board_routines.get_lichen_map(game_state, game_state.opp_player)

        for uid, nloc in Robot.next_locs.items():
            robot = self.robots[uid]

            if robot.is_mission(Mission.KAMIKAZE):
                continue

            # todo however be more aggressive if that's our primary ice?

            fight_lichen = opp_lichen[nloc] and robot.is_mission(Mission.SABOTAGE)
            fight_res = game_state.board.ice[nloc] or game_state.board.ore[nloc]
            our_primary_ice_sources = board_routines.get_ice_sources(game_state)

            if robot.loc != nloc and (fight_res or fight_lichen) and \
                    risk_map[nloc] > robot.weight > risk_map[robot.loc] and nloc == Robot.dst_locs.get(robot.id, None):
                # Wait, i.e. don't move... for how long?
                if board_routines.get_factory_neighborhood_map(game_state, game_state.opp_player)[nloc]:
                    # That's near the opponent factory, we can relax
                    p = 0.85
                elif board_routines.get_factory_neighborhood_map(game_state, game_state.player)[nloc] or nloc in our_primary_ice_sources.values():
                    # That's near our factory, be more aggressive
                    p = 0.2
                else:
                    # That's somewhere else
                    p = 0.6

                if np.random.random() < p:
                    actions_mod[uid] = []

        return actions_mod

    def apply_action_filter(self, actions):
        # Do not submit None actions (use RECHARGE action to cancel any other actions in the queue)
        # Also trim the list to max 20 individual actions
        actions = {uid: (al[:20] if isinstance(al, list) else al) for uid, al in actions.items() if al is not None}
        # Do not submit actions that are already the same as in the queue
        for uid, robot in self.robots.items():
            if uid in actions and robot.is_next_action_equiv(actions[uid]):
                del actions[uid]
        return actions

    def account_for_actions(self, game_state):
        """ Try to capture statistical efficiency of actions. This method may be computationally expensive while
        not useful in production. So don't call it by default. """

        mefs = self.mission_efficiency_stats

        n_light = len([1 for r in self.robots.values() if r.is_light()])
        n_heavy = len([1 for r in self.robots.values() if r.is_heavy()])

        # Log mission distribution at this step
        log.debug(f'@{game_state.real_env_steps} mission_distribution light: '
                  f'{Counter([r.get_mission() for r in self.robots.values() if r.is_light()])}')
        log.debug(f'@{game_state.real_env_steps} mission_distribution heavy: '
                  f'{Counter([r.get_mission() for r in self.robots.values() if r.is_heavy()])}')

        if self.game_state_prev:

            for uid, robot in self.robots.items():
                prev = self.units_prev.get(uid, None)
                unit = robot.unit
                self.units_prev[uid] = unit

                if not prev:
                    continue

                w = unit.weight

                # Exclude power pick-ups
                power_cost = prev.power - unit.power
                if power_cost < robot.weight:
                    power_cost = 0

                mn = robot.get_mission().name
                mefs[w][mn]['engagement'] += 1/n_light if robot.is_light() else 1/n_heavy

                if robot.is_mission(Mission.GO_ICE):
                    # We only count digging up, not trasferring to the factory
                    ice_diff = max(0, unit.cargo.ice - prev.cargo.ice)
                    mefs[w][mn]['value'] += ice_diff

                elif robot.is_mission(Mission.GO_ORE):
                    # We only count digging up, not trasferring to the factory
                    ore_diff = max(0, unit.cargo.ore - prev.cargo.ore)
                    mefs[w][mn]['value'] += ore_diff

                elif robot.is_mission(Mission.GRAZING) and unit.loc == prev.loc:
                    grazed = max(0, self.game_state_prev.board.rubble[unit.loc] - game_state.board.rubble[unit.loc])
                    mefs[w][mn]['value'] += grazed

                elif robot.is_mission(Mission.SABOTAGE) and unit.loc == prev.loc:
                    sabotaged = max(0, self.game_state_prev.board.lichen[unit.loc] - game_state.board.lichen[unit.loc])
                    mefs[w][mn]['value'] += sabotaged

                else:
                    continue

                # Exclude solar gain
                mefs[w][mn]['power_cost'] += power_cost + game_state.solar_gain[game_state.real_env_steps] * w
                mefs[w][mn]['time_cost'] += 1

            # Update efficiency
            for w in mefs.keys():
                for mn in mefs[w].keys():
                    mefs[w][mn]['power_eff'] = round(mefs[w][mn]['value'] / (1 + mefs[w][mn]['power_cost']), 3)
                    mefs[w][mn]['time_eff'] = round(mefs[w][mn]['value'] / (1 + mefs[w][mn]['time_cost']), 3)

        self.game_state_prev = game_state
