import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from a09_3a import board_routines, utils
from a09_3a.logger import log
from a09_3a.plant import Plant
from a09_3a.plant_manager import PlantManager
from a09_3a.robot import Robot, Mission


class MissionPlanner:

    def __init__(self, game_state, robot: Robot, home_plant: Plant = None):
        self.robot = robot
        self.robots = None
        self.home_plant = home_plant or PlantManager.get_instance().get_plant(robot.home_factory_id)
        self.n_steps_harassed = 0
        self.current_path = []
        self.current_cost = []

    def update_robots(self, robots):
        """ Not all missions need access to robots, but some do. """
        self.robots = robots

    def is_mission_accomplished(self, game_state):
        return False, None

    def is_too_far_from_home(self, game_state):
        """ Every mission should have a power breaker, that would allow the robot to go back and pick-up power,
        if only there's a plant with plenty of power nearby, and if it's not too late in the game.
        """
        # todo, we calculate the leash from any factory, but then we want to go home...
        r = self.robot
        leash = r.calculate_power_leash(game_state)
        leash2b = leash / r.battery
        end_game = 1000 - 300 * leash2b < game_state.real_env_steps
        # todo this should be updated only once in the robot.. scan for risk instead of harassment
        self.n_steps_harassed = max(0, self.n_steps_harassed + (1 if r.is_harassed(game_state) else -1))
        log.debug(f'\t{r.id} power: {r.power}, leash: {leash}, leash2b: {leash2b:.3f}, home plant power: '
                  f'{self.home_plant.power}, n steps harassed: {self.n_steps_harassed}')

        # todo this leash doesn't work well
        if (r.power2b() < 0.33 and r.power2b() < leash2b + 0.1 and
                (self.home_plant.has_plenty_of_power(game_state, r) or self.n_steps_harassed > 0) and
                not end_game):
            log.debug(f'{r.id} power leash activated')
            return True
        else:
            return False

    def has_destination(self, game_state):
        return Robot.dst_locs.get(self.robot.id, None)

    def select_destination(self, game_state):
        return None

    def travel_to_destination(self, game_state):
        r = self.robot
        # todo what if the robot wants to get to the factory... setting a single fixed dst can result in a deadlock
        dst_loc = Robot.dst_locs[r.id]
        if self.home_plant.is_factory_tile(dst_loc) and r.is_mission(Mission.GO_HOME):
            directions, cost, path_locs = r.find_way(game_state, destination_locs=self.home_plant.get_factory_tiles())
        else:
            directions, cost, path_locs = r.find_way(game_state, destination_locs=dst_loc)

        if not cost:
            log.debug(f'{r.id} the robot is at home already... but has extra actions in queue, do not interfere')
            return None, None

        assert cost

        cost[0] += r.unit.action_queue_cost(game_state)
        path_locs_str = f'{path_locs}' if len(path_locs) < 5 else f'{path_locs[0]} ..[{len(path_locs)}].. {path_locs[-1]}'
        log.debug(f'\t{r.id} loc {r.loc} finding way: {path_locs_str}, power: {r.unit.power}, cost: {sum(cost)}')

        # Update the path and apply the actions only if:
        # 1. The robot is not on the current path or
        # 2. The current path leads to some other destination or
        # 3. The cost of the new path is lower than the current cost
        if not self.current_path or r.loc != self.current_path[0] or \
                Robot.dst_locs[r.id] != self.current_path[-1] or sum(cost) < sum(self.current_cost[1:]):
            self.current_path = path_locs
            self.current_cost = cost
            actions = [r.unit.move(d) for d in directions]
            return actions, cost

        else:
            self.current_path.pop(0)
            self.current_cost.pop(0)
            return None, None

    def activity_at_destination(self, game_state):
        """ Return a list of actions. """
        return None, None

    def plan(self, game_state):
        r = self.robot
        # We need to update the home plant at every step, cause it could have changed..
        self.home_plant = PlantManager.get_instance().get_plant(r.home_factory_id)
        assert isinstance(self.home_plant, Plant)

        # 1. Check if mission accomplished, possibly assign a new mission
        done, new_mission = self.is_mission_accomplished(game_state)
        if done:
            r.mission_accomplished()
            if new_mission:
                r.set_mission(new_mission)
            return False

        # 2. Mission-specific power breaker.. todo the power leash should apply to transfer... but
        if not r.is_mission_any([Mission.BRING_WATER, Mission.GO_HOME, Mission.ATTACK,
                                                      Mission.KAMIKAZE, Mission.SUPPORT, Mission.SOLAR]) and \
                self.is_too_far_from_home(game_state):
            r.mission_accomplished()
            r.set_mission(Mission.GO_HOME)
            return False

        # 2.1 Opportunity hunt for selected (aggressive) missions
        if r.is_aggressive() and not r.is_mission(Mission.HUNT):
            # Opportunity hunt, when prey is easy to take down
            prey_id = r.scan_for_prey(game_state)
            if prey_id:
                log.debug(f'\t{r.id} found a prey, switch to hunt..')
                # Interruption, only for aggressive robots
                r.set_mission(Mission.HUNT)
                return False

        # 3. If we don't have a destination/target, select one
        if not self.has_destination(game_state):
            dst_loc = self.select_destination(game_state)
            if dst_loc:
                log.debug(f'\t{r.id} destination selected: {dst_loc}')
                # If the robot is heavy, it is possible the destination is currently taken by a light robot
                Robot.dst_locs[r.id] = dst_loc
            else:
                log.debug(f'\t{r.id} destination could not be selected, aborting the mission')
                # Shall we apply a score tab penalty to avoid re-selecting this mission?
                r.score_tab.set(r.get_mission(), -100.0, 20)
                r.mission_accomplished()
                return False

        # 4. If we are not at our destination/target, go there
        assert Robot.dst_locs[r.id]
        actions = None
        if r.loc != Robot.dst_locs[r.id]:
            al, _ = self.travel_to_destination(game_state)
            if al:
                actions = al
            else:
                log.debug(f'\t{r.id} has no power to travel, what to do?')

        # 5. Do whatever we need to do at the destination
        if r.loc == Robot.dst_locs[r.id] or actions:
            # Append actions to do at destination
            al, _ = self.activity_at_destination(game_state)
            if al is not None:
                actions = al if actions is None else actions + al

        return actions


class MissionPlannerAttack(MissionPlanner):

    def __init__(self, game_state, robot: Robot, target_fid, attstep):
        home_plant = PlantManager.get_instance().get_plant(robot.home_factory_id)
        super().__init__(game_state, robot, home_plant)
        # In case the factory is destroyed meanwhile, just store and use its location
        target_factory = game_state.factories[game_state.opp_player].get(target_fid, None)
        if target_factory:
            self.target_factory_loc = target_factory.loc
        else:
            log.warning(f'target factory has been destroyed: {target_fid}, rare situation')
            # Get another closest opponent's factory
            oflocs = board_routines.get_factory_locs(game_state, game_state.opp_player)
            loc_d = [(loc, utils.manhattan_distance(self.robot.loc, loc)) for loc in oflocs]
            loc_d = sorted(loc_d, key=lambda x: x[1])
            self.target_factory_loc = loc_d[0][0]
        self.attstep = attstep

    def is_mission_accomplished(self, game_state):
        if self.robot.is_light() and (game_state.real_env_steps > 992 or game_state.real_env_steps > 980 and
                                      self.robot.power2b() < 0.2):
            # Turn to kamikaze
            return True, Mission.KAMIKAZE

        return False, None

    def has_destination(self, game_state):
        r = self.robot
        step = game_state.real_env_steps

        if game_state.real_env_steps < self.attstep:
            _, cost, path = r.find_way(game_state, self.target_factory_loc, include_our_units=False)
            # Cut the path so it excludes the opp factory tiles
            cost = cost[:-3]
            power_aat = r.power - sum(cost)
            self.attstep = int(1000 - power_aat // r.avg_sabotage_cost - len(cost))

        # Select particular destination only when it's time
        if game_state.real_env_steps < self.attstep:
            log.debug(f'{r.id} ready to attack, steps to go: {self.attstep - game_state.real_env_steps}')
            # Just some temporary placeholder
            Robot.dst_locs[r.id] = self.home_plant.loc
            return True

        dst_loc = Robot.dst_locs.get(r.id, None)
        if dst_loc and game_state.board.lichen[dst_loc] > 0:
            return True
        else:
            Robot.dst_locs.pop(r.id, None)
            log.debug(f'{r.id} ATTACK!')
            return False

    def select_destination(self, game_state):

        # Take distance from opponents factories into account as well, only for units full of power!
        dmx_of = 10 - board_routines.DistanceMatrix.get_instance(self.target_factory_loc)

        # Find a new target tile
        opp_lichen_map = board_routines.get_lichen_map(game_state, game_state.opp_player) * \
                         np.clip(board_routines.DistanceMatrix.get_instance(self.target_factory_loc) < 10, 0, 1)

        # Target also areas not just individual tiles
        opp_lichen_map_conv = 0.05 * convolve2d(opp_lichen_map, utils.kernel_d2, mode='same')

        dmx_unit = np.clip(7 - board_routines.DistanceMatrix.get_instance(self.robot.loc), 0, 7)

        # Avoid squares under risk
        risk_ema = board_routines.BoardManager.get_instance().get_risk_map_ema()
        risk_cur = board_routines.get_risk_map(game_state)

        # Exclude taken locs
        taken_locs = list(Robot.dst_locs.values())
        opp_lichen = np.copy(opp_lichen_map)
        opp_lichen[tuple(np.array(taken_locs).T)] = 0

        if self.robot.is_heavy():
            opp_lichen_amt_bonus = 0.05 * opp_lichen - 2
        else:
            # Prefer tiles with little lichen
            opp_lichen_amt_bonus = 0.1 * (20 - opp_lichen)

        lmx = (dmx_of + 0.5 * dmx_unit + opp_lichen_amt_bonus + opp_lichen_map_conv - 0.5 * (risk_cur + risk_ema)) * np.clip(opp_lichen, 0, 1)

        loc = np.unravel_index(np.argmax(lmx), opp_lichen.shape)
        if lmx[loc] > 0:
            return loc
        else:
            if np.max(opp_lichen) == 0:
                log.debug(f'\t{self.robot.id}  opponent has no lichen')
            return None

    def travel_to_destination(self, game_state):

        if game_state.real_env_steps < self.attstep:
            # log.debug(f'{self.robot.id} ready to attack, steps to go: {game_state.real_env_steps - self.attstep}')
            return None, None

        dst_loc = Robot.dst_locs[self.robot.id]
        opp_lichen = board_routines.get_lichen_map(game_state, game_state.opp_player)
        assert dst_loc != self.robot.loc

        al, cl = [], []

        directions, cost, path_locs = self.robot.find_way(game_state, destination_locs=Robot.dst_locs[self.robot.id])
        assert cost

        cost[0] += self.robot.unit.action_queue_cost(game_state)
        path_locs_str = f'{path_locs}' if len(path_locs) < 5 else f'{path_locs[0]} ..[{len(path_locs)}].. {path_locs[-1]}'
        log.debug(f'\t{self.robot.id}  loc {self.robot.loc} finding way: {path_locs_str}, power: {self.robot.unit.power}, cost: {sum(cost)}')

        # Update the path and apply the actions only if:
        # 1. The robot is not on the current path or
        # 2. The current path leads to some other destination or
        # 3. The cost of the new path is lower than the current cost
        # if not self.current_path or self.robot.loc != self.current_path[0] or \
        #         Robot.dst_locs[self.robot.id] != self.current_path[-1] or sum(cost) < sum(self.current_cost[1:]):

        # Throw in BTW digs... what's the good threshold for grazing BTW? it used to be 0.2
        graze_btw = self.robot.is_heavy() and 0.2 < self.robot.power2b()

        if graze_btw and opp_lichen[self.robot.loc] > 20 and self.robot.loc != dst_loc:
            al += [self.robot.unit.dig(n=1)]
            cl += [self.robot.dc]

        for i, path_loc in enumerate(path_locs):
            al += [self.robot.unit.move(directions[i])]
            cl += [cost[i]]
            if graze_btw and opp_lichen[path_loc] > 0 and path_loc != dst_loc:
                al += [self.robot.unit.dig(n=1)]
                cl += [self.robot.dc]

        return al, cl

    def activity_at_destination(self, game_state):
        r = self.robot

        opp_lichen = board_routines.get_lichen_map(game_state, game_state.opp_player)
        n_digs = int(np.ceil(opp_lichen[r.loc] / r.unit.unit_cfg.DIG_LICHEN_REMOVED))

        home_plant = PlantManager.get_instance().get_plant(r.home_factory_id)
        l_dist = utils.manhattan_distance(r.loc, home_plant.loc)

        if game_state.real_env_steps < 960 - 2 * l_dist:
            leash = r.calculate_power_leash(game_state)
            # Limit the digs by available power
            n_digs = min(n_digs, (r.power - leash - r.battery//20) // r.dc)

        actions = [r.unit.dig(n=1)] * n_digs
        cost = [r.dc] * n_digs
        return actions, cost


class MissionPlannerBringWater(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant, dst_plant_id: str):
        super().__init__(game_state, robot, home_plant)
        self.dst_plant_id = dst_plant_id
        # The mission has a few stages:
        # 0 -- need to go to home_factory to pick-up water and power
        # 1 -- travel to the destination factory
        # 2 -- drop the cargo
        self.stage = 0

    def is_mission_accomplished(self, game_state):

        log.debug(f'\t{self.robot.id} from {self.home_plant.id} to {self.dst_plant_id}')
        dst_plant = PlantManager.get_instance().get_plant(self.dst_plant_id)
        if dst_plant and self.stage == 1 and self.robot.unit.cargo.water == 0:
            # Mission is succesfull, when we unload water at destination factory
            log.debug(f'\t{self.robot.id} water brought sucessfully')
            Robot.stats['water_brought'] += 1
            return True, None
        elif dst_plant is None:
            log.debug(f'\t{self.robot.id} destination plant {self.dst_plant_id} has vanished, aborting the mission')
            Robot.stats['water_toolate'] += 1
            return True, None
        else:
            return False, None

    def select_destination(self, game_state):
        assert self.dst_plant_id != self.home_plant.id
        if self.stage == 0:
            if self.home_plant.is_factory_tile(self.robot.loc):
                # Arrived at the home factory...
                if self.robot.unit.cargo.water > 0:
                    # We already have water loaded, go to stage 1
                    self.stage = 1
                else:
                    return self.robot.loc
            else:
                # Go to our home_plant to pick up water
                _, _, path_locs = self.robot.find_way(game_state, destination_locs=self.home_plant.get_factory_tiles())
                assert path_locs
                return path_locs[-1]

        if self.stage == 1:
            dst_plant = PlantManager.get_instance().get_plant(self.dst_plant_id)
            # Going to the destination factory
            if dst_plant.is_factory_tile(self.robot.loc):
                # Arrived at destination factory, don't change the destination
                return self.robot.loc
            else:
                _, _, path_locs = self.robot.find_way(game_state, destination_locs=dst_plant.get_factory_tiles())
                assert path_locs
                return path_locs[-1]

    def activity_at_destination(self, game_state):
        """ Return a tuple: list of actions, and a list of cost. """
        if self.stage == 0:
            if self.home_plant.is_factory_tile(self.robot.loc):
                # Reset destination, so that next turn we reselect
                Robot.dst_locs.pop(self.robot.id, None)

            # Load water
            actions = [self.robot.unit.pickup(pickup_resource=utils.ResourceId.WATER.value,
                                              pickup_amount=min(self.home_plant.cargo.water//2, 100))]
            cost = [self.robot.weight]
            return actions, cost

        if self.stage == 1:
            # Unload water
            actions = [self.robot.unit.transfer(0, transfer_resource=utils.ResourceId.WATER.value,
                                                transfer_amount=self.robot.unit.cargo.water)]
            cost = [self.robot.weight]
            return actions, cost


class MissionPlannerGoHome(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot, home_plant)
        self.step0 = game_state.real_env_steps

    def is_mission_accomplished(self, game_state):
        is_ft = self.home_plant.is_factory_tile(self.robot.loc)
        queue_empty = len(self.robot.unit.action_queue) == 0
        return is_ft and queue_empty, None

    def has_destination(self, game_state):
        # Refresh every 5 steps
        if (game_state.real_env_steps - self.step0) % 5 == 0:
            return None
        else:
            return Robot.dst_locs.get(self.robot.id, None)

    def select_destination(self, game_state):
        home_tiles = self.home_plant.get_factory_tiles()
        _, _, path_locs = self.robot.find_way(game_state, destination_locs=home_tiles)
        return path_locs[-1] if path_locs else None

    def activity_at_destination(self, game_state):
        r = self.robot

        # Unload cargo if any, also charge the primary robot unconditionally
        actions = []

        if r.unit.cargo.ice:
            actions += [r.unit.transfer(0, transfer_resource=utils.ResourceId.ICE.value, transfer_amount=r.unit.cargo.ice)]

        if r.unit.cargo.ore:
            actions += [r.unit.transfer(0, transfer_resource=utils.ResourceId.ORE.value, transfer_amount=r.unit.cargo.ore)]

        return actions, None


class MissionPlannerGoResource(MissionPlanner):
    def __init__(self, game_state, robot: Robot, home_plant: Plant, res_int: int):
        super().__init__(game_state, robot, home_plant)
        self.robots = None
        self.supply_in_action = 0
        self.res_int = res_int

    def travel_to_destination(self, game_state):
        r = self.robot

        # If the unit is heavy and in fight over ice tile which is next to the opponents factory, and it is
        # waiting next to the ice tile which is away from the factory and not under risk, and there are no other clear
        # tiles to make it easier to wrangle, take this opportunity to clear the tile from the rubble
        if r.is_heavy() and utils.is_adjacent(r.loc, Robot.dst_locs.get(r.id, r.loc)) and r.is_idle() and \
                game_state.board.rubble[r.loc] > 0:

            risk_map = board_routines.get_risk_map(game_state)
            ofnm = board_routines.get_factory_neighborhood_map(game_state, game_state.opp_player)

            if risk_map[r.loc] < r.pweight and ofnm[r.loc] == 0 and ofnm[Robot.dst_locs[r.id]] == 1:
                n_digs = int(np.ceil(game_state.board.rubble[r.loc] / r.unit_cfg.DIG_RUBBLE_REMOVED))
                if n_digs > 0 and r.power > n_digs * r.unit.dig_cost(game_state) + r.battery//15:
                    actions = [r.unit.dig(n=n_digs)]
                    al, _ = super().travel_to_destination(game_state)
                    actions += al if al else []
                    return actions, None

        return super().travel_to_destination(game_state)

    def activity_at_destination(self, game_state):
        # Just try to do as many digs a possible within the leash, schedule minimum 3 digs
        r = self.robot
        assert self.robots

        # The chain has to be in place
        dst_loc = Robot.dst_locs[r.id]

        if r.loc == dst_loc and not r.chain_path:
            r.establish_chain(game_state, self.robots)
        elif r.loc != dst_loc:
            r.chain_path = None

        if r.is_chain_locked(self.robots):
            assert r.chain_path
            if not r.is_idle() and self.supply_in_action % 20 != 0:
                self.supply_in_action += 1
                return None, None
            else:
                loc = r.chain_path[0]
                d_up = utils.direction_to(dst_loc, loc)
                surplus = int(np.round(np.clip((r.power - 2400) // 100, 0, 6)))

                if self.home_plant.cargo.water < 100 and self.res_int == utils.ResourceId.ICE.value or \
                        self.home_plant.power > 5000:
                    # Intensive pace
                    actions = [r.unit.move(0, repeat=0, n=1),
                               r.unit.transfer(d_up, self.res_int, 75, repeat=1, n=1),
                               r.unit.dig(repeat=3, n=3)]
                    self.robot.set_power_needed_per_step(39 - surplus)

                elif self.home_plant.cargo.water > 150 and self.home_plant.power < 600 and \
                        self.home_plant.cargo.metal > 100:
                    # Slow down a bit, to let the power build in the factory to produce robots
                    actions = [r.unit.move(0, repeat=0, n=1),
                               r.unit.transfer(d_up, self.res_int, 50, repeat=2, n=2),
                               r.unit.dig(repeat=1, n=1)]
                    self.robot.set_power_needed_per_step(14 - surplus)

                elif self.home_plant.cargo.water > 250 and self.home_plant.power < 2500 and \
                        self.res_int == utils.ResourceId.ICE.value:
                    # Slow down with ice a bit, to let the power build up for ore mining
                    actions = [r.unit.move(0, repeat=0, n=1),
                               r.unit.transfer(d_up, self.res_int, 50, repeat=2, n=2),
                               r.unit.dig(repeat=1, n=1)]
                    self.robot.set_power_needed_per_step(14 - surplus)

                else:
                    actions = [r.unit.move(0, repeat=0, n=1),
                               r.unit.transfer(d_up, self.res_int, 50, repeat=1, n=1),
                               r.unit.dig(repeat=1, n=1)]
                    self.robot.set_power_needed_per_step(24 - surplus)

                self.supply_in_action += 1
                return actions, None

        else:
            self.supply_in_action = 0
            # Limit the digs by available power
            leash = r.calculate_power_leash(game_state)
            n_digs = (r.power - leash - r.battery // 20) // r.dc

            if n_digs > 2:
                actions = [r.unit.dig(n=n_digs)]
                cost = [r.dc * n_digs]
                return actions, cost
            elif n_digs >= 0:
                return None, None
            else:
                # Robot is just waiting for more power... how to make the location available?
                # Switch to recharge?
                # Robot.dst_locs.pop(r.id, None)
                # raise MissionOnTop(Mission.RECHARGE, recharge_to=r.power + 10*r.dc)
                return [], []


class MissionPlannerGoIce(MissionPlannerGoResource):

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot, home_plant, utils.ResourceId.ICE.value)
        self.step_0 = game_state.real_env_steps

    def is_mission_accomplished(self, game_state):

        r = self.robot
        steps_elapsed = game_state.real_env_steps - self.step_0

        leash = r.calculate_power_leash(game_state)
        plant_enough_power = self.home_plant.power > r.battery - r.power + (500 if r.is_light() else 0)
        is_loaded = r.unit.cargo.ice + r.unit.cargo.ore >= 60 * r.weight
        is_full = r.unit.cargo.ice + r.unit.cargo.ore == r.unit.unit_cfg.CARGO_SPACE
        end_game = game_state.real_env_steps > 930
        priority = r.is_primary_ice and self.home_plant.power > r.battery - r.power
        n_digs = (r.power - leash - r.battery // 20) // r.dc
        is_in_chain = r.is_chain_locked(self.robots)

        log.debug(f'\t{r.id} power: {r.power}, n_digs: {n_digs}, leash: {leash}, plant power: '
                  f'{self.home_plant.power}, cargo: {r.unit.cargo.ice + r.unit.cargo.ore}, steps elapsed: '
                  f'{steps_elapsed}, checking for mission accomplished..')

        if is_full or (n_digs <= 0 or n_digs < 3 and r.is_idle()) and not is_in_chain and \
                (plant_enough_power or priority or is_loaded or end_game or self.home_plant.cargo.water < 100):
            log.debug(f'\t{r.id} yes, normal condition')
            return True, Mission.GO_HOME

        # Another case to accomplish mission to check for ORE opportunity is when the factory has more than 200 water
        # and we want to get some ore
        if self.home_plant.cargo.water > 200 and r.is_primary_ice and r.is_primary_ore and \
                steps_elapsed > 21 + game_state.real_env_steps//15:
            log.debug(f'\t{r.id} yes, ore check')
            return True, Mission.GO_ORE

        # By the end game, if the factory has enough water, break
        if game_state.real_env_steps > 920 and (self.home_plant.cargo.water > 400 or self.home_plant.water_balance > 150):
            log.debug(f'\t{r.id} yes, end game condition')
            # Turn off any primary ice role
            r.is_primary_ice = False
            return True, None

        return False, None

    def select_destination(self, game_state):
        score, dst_loc = self.robot.evaluate_mission_go_res(game_state, self.home_plant, resource='ice')
        # We can abort the mission if the actual score is below threshold
        if score > 0.3 or self.robot.is_primary_ice:
            return dst_loc
        else:
            log.debug(f'\t{self.robot.id} score {score} too low, aborting the mission...')
            return None


class MissionPlannerGoOre(MissionPlannerGoResource):

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot, home_plant, utils.ResourceId.ORE.value)

    def is_mission_accomplished(self, game_state):
        """ The mission GO_ORE is accomplished when the robot has no more spare power to dig, but the factory has enough
        power to recharge it.
        """
        r = self.robot
        leash = r.calculate_power_leash(game_state)
        plant_enough_power = self.home_plant.power > r.battery - r.power + (500 if r.is_light() else 0)
        is_loaded = r.unit.cargo.ice + r.unit.cargo.ore >= 50 * r.weight
        is_full = r.unit.cargo.ice + r.unit.cargo.ore == r.unit.unit_cfg.CARGO_SPACE
        end_game = game_state.real_env_steps > 800
        opening = game_state.real_env_steps < 200
        n_digs = (r.power - leash - r.battery // 20) // r.dc
        is_in_chain = r.is_chain_locked(self.robots)

        if is_full or (n_digs <= 0 or n_digs < 3 and r.is_idle()) and not is_in_chain and \
                (plant_enough_power or is_loaded or end_game or opening):
            log.debug(f'\t{r.id} power: {r.power}, n_digs: {n_digs}, leash: {leash}, plant power: '
                      f'{self.home_plant.power}, cargo: {r.unit.cargo.ice + r.unit.cargo.ore}, GO_HOME pickup '
                      f'some power and/or dump the cargo')
            return True, Mission.GO_HOME

        if r.is_chain_locked(self.robots) and end_game:
            return True, None

        return False, None

    def select_destination(self, game_state):
        score, dst_loc = self.robot.evaluate_mission_go_res(game_state, self.home_plant, resource='ore')
        # We can abort the mission if the actual score is below threshold
        if score > 0.3:
            return dst_loc
        else:
            log.debug(f'\t{self.robot.id}  score {score} too low, aborting the mission...')
            return None

    def travel_to_destination(self, game_state):
        r = self.robot
        actions, cost = [], []
        if self.home_plant.is_factory_tile(r.loc) and \
                utils.manhattan_distance(self.home_plant.loc, Robot.dst_locs[r.id]) > 3 and \
                r.power2b() < 0.75 and self.home_plant.power > r.battery - r.unit.power:
            power_to_pick = min(self.home_plant.power, r.battery - r.unit.power)
            actions += [r.unit.pickup(utils.ResourceId.POWER.value, power_to_pick, repeat=0, n=1)]

        al, _ = super().travel_to_destination(game_state)
        actions = al if al is None else actions + al
        return actions, None


class MissionPlannerGraze(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot, home_plant)

    def is_mission_accomplished(self, game_state):
        dst_loc = Robot.dst_locs.get(self.robot.id, None)

        if self.robot.power2b() < 0.1:
            return True, Mission.RECHARGE

        if dst_loc and game_state.board.rubble[dst_loc] == 0:
            log.debug(f'\t{self.robot.id}  tile {dst_loc} has been cleared, mission accomplished')
            return True, None
        else:
            return False, None

    def select_destination(self, game_state):

        # todo if a RES tile is a bit away, maybe dig out a corridor

        # Select the grazing target, not too much rubble, close to the home factory, connected with the
        # lake and not taken yet
        fdm = np.copy(PlantManager.get_instance().get_dominance_map(self.home_plant.id))
        # Include factory tiles, for now
        fdm[tuple(self.home_plant.get_factory_tiles().T)] = 1

        # On the dominance map, only the clear tiles without rubble but dominant proximity of the factory are 1
        # All other tiles are marked with distances from the factory
        fdm_ext = np.sign(convolve2d(fdm, utils.kernel_d1, mode='same'))
        grazing_exclusion_map = PlantManager.get_grazing_exclusion_map(game_state)

        # Take the distance from the factory and from the unit into account as well
        # Distance from the factory way more important, f(1:2:12) = [ 7.875,  6.875,  4.875,  1.875, -2.125, -7.125]
        d_fac = 8 * (1 - (board_routines.DistanceMatrix.get_instance(self.home_plant.loc) / 8) ** 2)
        d_loc = 10 - board_routines.DistanceMatrix.get_instance(self.robot.loc)

        tadv = board_routines.get_territorial_advantage(game_state)

        # Light robots should prefer rubble < 20, heavies should prefer heavy rubble
        rubble = game_state.board.rubble
        if self.robot.is_light():
            # f(5: 10:95) = [ 4.4,  3.4,  2.4,  1.4,  0.4, -0.6, -1.6, -2.6, -3.6, -4.6, -5.6]
            rubble_score = 5 - ((rubble + 1) // 2 / self.robot.dc)
        else:
            # A non-linear score function for heavy robots.. f(5:10:95) = [1,  5,  3,  3,  1,  1, -1, -1, -3, -3, -5]
            rubble_score = 2 * (4 - np.ceil(rubble / 20).astype(np.int) - np.where(rubble < 14, 2, 0)) - 3

        score = (d_fac + 1.0 * d_loc + rubble_score + 5 + 5 * tadv) * (1 - grazing_exclusion_map) * \
                np.clip(rubble, 0, 1) * fdm_ext

        # Exclude locations taken
        for uid_, loc_ in Robot.dst_locs.items():
            if loc_ is not None and uid_ != self.robot.id:
                score[loc_] = 0

        # Density
        score_density = convolve2d(score, utils.kernel_d2, mode='same')
        score += score_density * (1 - grazing_exclusion_map) * np.clip(game_state.board.rubble, 0, 1)

        # Choose the location with maximum score
        dst_loc = np.unravel_index(np.argmax(score), score.shape)

        log.debug(f'\t{self.robot.id} grazing best location: {dst_loc}, score: {score[dst_loc]:.2f}')

        if score[dst_loc] > 0:
            return dst_loc
        else:
            # No tiles for grazing found
            log.debug(f'\t{self.robot.id} not efficient enough, mission accomplished')
            return None

    def activity_at_destination(self, game_state):
        r = self.robot
        dst_loc = Robot.dst_locs[r.id]

        # We checked in is_mission_accomplished() the rubble is there...
        assert game_state.board.rubble[dst_loc] > 0

        # Do not dig beyond the power leash
        leash = r.calculate_power_leash(game_state)
        n_digs_req = int(np.ceil(game_state.board.rubble[dst_loc] / r.unit_cfg.DIG_RUBBLE_REMOVED))
        # Limit the digs by available power
        n_digs_can = (r.power - leash - r.battery//20) // r.dc
        if n_digs_can >= min(n_digs_req, 3):
            actions = [r.unit.dig(n=min(n_digs_req, n_digs_can))]
            cost = [r.dc * min(n_digs_req, n_digs_can)]
        elif n_digs_can >= 0:
            actions = None
            cost = None
        else:
            actions = []
            cost = []

        log.debug(f'\t{r.id} loc: {r.loc}, dst_loc: {dst_loc}, power: {r.unit.power}, n_digs_req: {n_digs_req}, '
                  f'n_digs_can: {n_digs_can}, cost: {cost}')

        return actions, cost


class MissionPlannerGuard(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot, home_plant)
        self.step0 = game_state.real_env_steps

    def is_mission_accomplished(self, game_state):
        r = self.robot
        log.debug(f'\t{r.id} dst_loc: {Robot.dst_locs.get(r.id, None)}, prey: {Robot.preys_targeted.get(r.id, None)}')
        go_home_score = (utils.low_pass(r.power2b(), 0.2, 0.5) *
            utils.hi_pass(self.home_plant.power, 4000, 4000 + r.battery) *
            utils.low_pass(utils.manhattan_distance(r.loc, self.home_plant.loc), 5, 25) *
            utils.low_pass(game_state.real_env_steps, 940, 1000) *
            utils.hi_pass(game_state.real_env_steps, 500, 800))
        if go_home_score > 0.33:
            # Go home pick-up power
            return True, Mission.GO_HOME

        # Guarding robots are re-evaluated from time to time...
        condition = np.random.random() < 0.01
        return condition, None

    def select_destination(self, game_state):
        """ Pick an area to guard, it should be:
        1. Where we have plenty of lichen
        2. The risk is balanced, i.e. our light engage opponent's light, etc.
        3. Choose safer areas for units with low power, more exposed for units with plenty of power
        4. Distribute our units evenly
        """

        udm = 2 - board_routines.get_units_density_map(game_state, game_state.player)
        udm -= 2 * board_routines.get_factory_neighborhood_map(game_state, game_state.opp_player)

        # Not too far and not too close from the home plant, a ring of radius 4 is optimal.. depends also on how
        # many robots are around
        dmx_hp = 4 - np.abs(4 - board_routines.DistanceMatrix.get_instance(self.home_plant.loc))

        # Do not clog up our factories
        dmx_hp[self.home_plant.pos_slice] = -100
        dmx_hp[tuple(self.home_plant.get_connection_tiles().T)] = -10

        # Stay away from opponent's territorial adantage
        tadv = np.clip(board_routines.get_territorial_advantage(game_state), 0, 1)
        # Correlate tadv 0 with strong robots, and tadv 1 with weak robots
        # -1  -1  -> 0
        #  0  -1  -> 0
        #  1  -1  -> 1
        # -1   0  -> 0
        #  0   0  -> 1
        #  1   0  -> 0
        # -1   1  -> 1
        #  0   1  -> 0
        #  1   1  -> 0
        tadv_score = 3 * (1 - abs(tadv + (2 * self.robot.power2b() - 1)))

        # Don't walk far (but only if we're near our home plant in the first place)
        if utils.manhattan_distance(self.robot.loc, self.home_plant.loc) < 10:
            dmx_us = -board_routines.DistanceMatrix.get_instance(self.robot.loc)
        else:
            dmx_us = 0

        # Light robots avoid heavy,
        risk_map_cur = board_routines.get_risk_map(game_state)
        risk_score = np.clip(self.robot.pweight - risk_map_cur, -3, 1)

        our_lichen_map = board_routines.get_lichen_map(game_state, game_state.player)
        lichen_density = 0.05 * convolve2d(our_lichen_map, utils.kernel_d2, mode='same')

        score = dmx_hp + 1.5*dmx_us + 2 * udm + tadv_score - lichen_density + risk_score

        # Choose the location with the maximum score
        dst_loc = np.unravel_index(np.argmax(score), score.shape)
        return dst_loc


class MissionPlannerHunt(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant = None, prey_id: str = None):
        super().__init__(game_state, robot, home_plant)
        Robot.preys_targeted[self.robot.id] = prey_id

    def is_mission_accomplished(self, game_state):
        prey_id = Robot.preys_targeted.get(self.robot.id, None)
        log.debug(f'\t{self.robot.id} prey id: {prey_id}')

        if prey_id and prey_id in game_state.units[game_state.opp_player]:
            prey = game_state.units[game_state.opp_player][prey_id]
            opp_fdm = board_routines.get_factories_distance_map(game_state, game_state.opp_player)
            l_dist = utils.manhattan_distance(self.robot.loc, prey.loc)
            # Also give up, when a strong prey gets away more than 8 tiles from our factory or into its territory
            our_fdm = board_routines.get_factories_distance_map(game_state, game_state.player)
            tadv = board_routines.get_territorial_advantage(game_state)

            vulnerability = opp_fdm[prey.loc] / 100 - prey.power / prey.unit_cfg.BATTERY_CAPACITY
            if our_fdm[prey.loc] > 15 or ((opp_fdm[prey.loc] < l_dist or our_fdm[prey.loc] > 8 or
                (our_fdm[prey.loc] > 5 and tadv[prey.loc] < 0.0)) and vulnerability < 0.0):
                log.debug(f'\t{self.robot.id} prey ({prey_id}) got away, ofdm: {opp_fdm[prey.loc]}, l_dist: {l_dist},'
                          f'vulnerability: {vulnerability:.3f}, aborting HUNT mission')
                return True, None

        elif prey_id:
            log.debug(f'\t{self.robot.id} it seems our prey ({prey_id}) has been destroyed')
            Robot.stats['hunts_completed'] += 1
            return True, None

        return False, None

    def has_destination(self, game_state):
        # On a hunt we update the destination loc at every turn
        Robot.dst_locs.pop(self.robot.id, None)
        return False

    def select_destination(self, game_state):
        prey_id = Robot.preys_targeted.get(self.robot.id, None)

        if not prey_id:
            prey_id = self.robot.scan_for_prey(game_state)

        if prey_id:
            # Return prey's location
            prey = game_state.units[game_state.opp_player][prey_id]
            Robot.preys_targeted[self.robot.id] = prey_id
            return prey.loc

        return None


class MissionPlannerKamikaze(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot)

    def has_destination(self, game_state):
        if game_state.real_env_steps == 999:
            return False

        dst_loc = Robot.dst_locs.get(self.robot.id, None)
        return dst_loc is not None and game_state.board.lichen[dst_loc] > 10

    def select_destination(self, game_state):

        # todo heavy units, also can wait to be destroyed on lichen squares
        opp_lichen = board_routines.get_lichen_map(game_state, game_state.opp_player)
        if game_state.real_env_steps == 999 and opp_lichen[self.robot.loc] > 0:
            return self.robot.loc

        steps_left = 999 - game_state.real_env_steps
        rng = min(steps_left, max(0, self.robot.power // 1))

        udmx = board_routines.DistanceMatrix.get_instance(self.robot.loc)
        unit_range = (udmx < max(1, rng)).astype(np.int) * ((rng - udmx) / 10 + 1)

        lmx = unit_range * opp_lichen

        # Exclude taken locs
        taken_locs = list(Robot.dst_locs.values())
        lmx[tuple(np.array(taken_locs).T)] /= 5

        loc = np.unravel_index(np.argmax(lmx), lmx.shape)
        if lmx[loc] > 0:
            return loc
        else:
            log.debug(f'\t{self.robot.id} opponent has no lichen to kamikaze in our range.. '
                      f'select the closest tile that does not have our lichen')

            self.robot.score_tab.set(Mission.SABOTAGE, -100, 10)
            return None

    def activity_at_destination(self, game_state):
        actions = [self.robot.unit.self_destruct(repeat=1)]
        cost = [self.robot.unit.unit_cfg.SELF_DESTRUCT_COST]
        return actions, cost


class MissionPlannerRecharge(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant, recharge_to: int = 0):
        super().__init__(game_state, robot, home_plant)
        self.recharge_to = recharge_to
        self.step0 = game_state.real_env_steps

    def is_mission_accomplished(self, game_state):
        step = game_state.real_env_steps
        if 0 < self.recharge_to <= self.robot.power or 0.9 < self.robot.power2b() or \
                self.step0 + 50 < step or 1000 - step < 120 * self.robot.power2b():
            return True, None

        # Preventing light robots from picking-up power doesn't seem to work well
        #  but for now just prevent lights from going home to recharge
        u2 = utils.hi_pass(self.home_plant.power - self.robot.battery, 1000, 7000)
        u3 = utils.low_pass(utils.manhattan_distance(self.robot.loc, self.home_plant.loc), 1, 20)
        u4 = utils.low_pass(self.robot.power2b(), 0.3, 0.4) * utils.hi_pass(self.robot.power2b(), 0.1, 0.2)
        # If we direct many low-power robots towards the factory, it will become cramped

        if u2 * u3 * u4 > 0.5**3:
            log.debug(f'\t{self.robot.id} ... instead of waiting-recharging, GO_HOME and pickup power')
            return True, Mission.GO_HOME

        return False, None

    def has_destination(self, game_state):
        return False

    def select_destination(self, game_state):
        # We actually checked if we should move somewhere in has_destination(), if it returned None, then it means
        # we don't need to go anywhere, so stay put, but don't return None as it would terminate the mission.
        ofmap = board_routines.get_factory_map(game_state, game_state.player)
        loc = self.robot.loc
        if ((game_state.board.ice[loc] or game_state.board.ore[loc]) and
                loc in Robot.dst_locs.values()) or ofmap[loc]:

            # Avoid opponent's territory
            tadv = board_routines.get_territorial_advantage(game_state)
            # Avoid our factory and the connection tiles
            our_fnm = board_routines.get_factory_neighborhood_map(game_state, game_state.player)
            our_fdm = 3 - np.abs(3 - board_routines.get_factories_distance_map(game_state, game_state.player))

            unit_dmx = np.clip(7 - board_routines.DistanceMatrix.get_instance(loc), 0, 7)
            res_free = 1 - (game_state.board.ice + game_state.board.ore)

            score_map = 2 * unit_dmx + 5 * tadv + 3 * res_free + 3 * (1 - our_fnm) + our_fdm

            # Avoid the plant tiles, especially the center
            score_map *= (1 - ofmap)

            dst_locs = np.argwhere(score_map == np.max(score_map))
            # Find out which is the least costly to go
            _, _, path_locs = self.robot.find_way(game_state, destination_locs=dst_locs)
            if path_locs:
                return path_locs[-1]
            else:
                log.debug(f'\t{self.robot.id} Could not find a recharging spot...')

        # Return a "nowhere to go" destination, only self.travel_to_destination() will understand
        return -1, -1

    def travel_to_destination(self, game_state):
        if Robot.dst_locs[self.robot.id] == (-1, -1):
            # We don't have to go anywhere
            # return (None, None) if self.robot.is_idle() else ([], None)
            # Empty action list is automatically removed if the robot is idle.. so no waste
            return [], []
        else:
            return super().travel_to_destination(game_state)


class MissionPlannerSabotage(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot, home_plant)
        self.targeting_res = False

    def has_destination(self, game_state):
        dst_loc = Robot.dst_locs.get(self.robot.id, None)
        if dst_loc and (game_state.board.lichen[dst_loc] > 0 or self.targeting_res):
            return True
        else:
            Robot.dst_locs.pop(self.robot.id, None)
            return False

    def select_destination(self, game_state):

        # todo target also water supplies or supply chains

        self.targeting_res = False

        if self.robot.is_heavy():
            # Try to target water or supply chains
            # Only for factories that are separated from opponent's plants
            opf_locs = board_routines.get_factory_locs(game_state, game_state.opp_player)
            for oppfid, oppf in game_state.factories[game_state.opp_player].items():
                if oppf.cargo.water < 300 and utils.manhattan_distance(self.robot.loc, oppf.loc) < 15:
                    ice_dmx = game_state.board.ice * (board_routines.DistanceMatrix.get_instance(oppf.loc) < 7).astype(np.int)
                    ice_locs = list(map(tuple, np.argwhere(ice_dmx)))
                    if 0 < len(ice_locs) <= len([1 for l in opf_locs if utils.manhattan_distance(l, oppf.loc) < 9]):
                        for ice_loc in ice_locs:
                            if ice_loc not in Robot.dst_locs.values():
                                log.debug(f'{self.robot.id} targeting opponent ice source: {ice_loc}')
                                self.targeting_res = True
                                return ice_loc

            for oppfid, oppf in game_state.factories[game_state.opp_player].items():
                if utils.manhattan_distance(self.robot.loc, oppf.loc) < 15:
                    ore_dmx = game_state.board.ore * (board_routines.DistanceMatrix.get_instance(oppf.loc) < 7).astype(np.int)
                    ore_locs = list(map(tuple, np.argwhere(ore_dmx)))
                    if 0 < len(ore_locs) <= len([1 for l in opf_locs if utils.manhattan_distance(l, oppf.loc) < 9]):
                        for ore_loc in ore_locs:
                            if ore_loc not in Robot.dst_locs.values():
                                log.debug(f'{self.robot.id} targeting opponent ore source: {ore_loc}')
                                self.targeting_res = True
                                return ore_loc

        if game_state.real_env_steps >= 800:
            # Target lichen
            # Find a new target tile
            pm = PlantManager.get_instance()
            opp_lichen_map = board_routines.get_lichen_map(game_state, game_state.opp_player)

            # Target also areas not just individual tiles
            opp_lichen_map_conv = 0.05 * convolve2d(opp_lichen_map, utils.kernel_d2, mode='same')

            # Take distance from opponents factories into account as well, only for units full of power!
            dmx_of = 10 + np.zeros(shape=board_routines.DistanceMatrix.get_shape())
            for oid, oplant in pm.opp_plants.items():
                dmx_of = np.minimum(dmx_of, board_routines.DistanceMatrix.get_instance(oplant.loc))
            dmx_of_score = 10 - dmx_of

            dmx_unit = 10 - board_routines.DistanceMatrix.get_instance(self.robot.loc)

            # Avoid squares under risk
            risk_ema = board_routines.BoardManager.get_instance().get_risk_map_ema()
            risk_cur = board_routines.get_risk_map(game_state)

            # Exclude taken locs
            taken_locs = list(Robot.dst_locs.values())
            opp_lichen = np.copy(opp_lichen_map)
            opp_lichen[tuple(np.array(taken_locs).T)] = 0

            if self.robot.is_heavy():
                opp_lichen_amt_bonus = 0.05 * opp_lichen - 2
            else:
                # Prefer tiles with little lichen
                opp_lichen_amt_bonus = 0.1 * (20 - opp_lichen)

            lmx = (0.5 * dmx_unit + opp_lichen_amt_bonus + opp_lichen_map_conv + dmx_of_score -
                   0.5 * (risk_ema + risk_cur)) * np.clip(opp_lichen, 0, 1)

            loc = np.unravel_index(np.argmax(lmx), opp_lichen.shape)
            if lmx[loc] > 0:
                # Check if the mission/destination is viable
                _, cost, _ = self.robot.find_way(game_state, destination_locs=[loc])
                if self.robot.power < 2 * sum(cost) or game_state.real_env_steps + 2 * len(cost) > 999:
                    log.debug(f'\t {self.robot.id} mission inefficient, returning destination None, i.e. aborting')
                    return None
                else:
                    return loc
            else:
                if np.max(opp_lichen) == 0:
                    log.debug(f'\t{self.robot.id}  opponent has no lichen')
                return None

        log.debug(f'\t{self.robot.id} could not target neither water nor lichen')
        return None

    def travel_to_destination(self, game_state):

        dst_loc = Robot.dst_locs[self.robot.id]
        opp_lichen = board_routines.get_lichen_map(game_state, game_state.opp_player)
        assert dst_loc != self.robot.loc

        al, cl = [], []

        directions, cost, path_locs = self.robot.find_way(game_state, destination_locs=Robot.dst_locs[self.robot.id])
        assert cost

        cost[0] += self.robot.unit.action_queue_cost(game_state)
        path_locs_str = f'{path_locs}' if len(path_locs) < 5 else f'{path_locs[0]} ..[{len(path_locs)}].. {path_locs[-1]}'
        log.debug(f'\t{self.robot.id}  loc {self.robot.loc} finding way: {path_locs_str}, power: {self.robot.unit.power}, cost: {sum(cost)}')

        # Update the path and apply the actions only if:
        # 1. The robot is not on the current path or
        # 2. The current path leads to some other destination or
        # 3. The cost of the new path is lower than the current cost
        # if not self.current_path or self.robot.loc != self.current_path[0] or \
        #         Robot.dst_locs[self.robot.id] != self.current_path[-1] or sum(cost) < sum(self.current_cost[1:]):

        # Throw in BTW digs.. what's the good treshold for grazing BTW it used to be 0.2
        graze_btw = self.robot.is_heavy() and 0.0 < self.robot.power2b()

        if graze_btw and opp_lichen[self.robot.loc] > 0 and self.robot.loc != dst_loc:
            al += [self.robot.unit.dig(n=1)]
            cl += [self.robot.dc]

        for i, path_loc in enumerate(path_locs):
            al += [self.robot.unit.move(directions[i])]
            cl += [cost[i]]
            if graze_btw and opp_lichen[path_loc] > 0 and path_loc != dst_loc:
                al += [self.robot.unit.dig(n=1)]
                cl += [self.robot.dc]

        return al, cl

    def activity_at_destination(self, game_state):
        r = self.robot

        if self.targeting_res:
            return [], None

        else:
            opp_lichen = board_routines.get_lichen_map(game_state, game_state.opp_player)
            n_digs = int(np.ceil(opp_lichen[r.loc] / r.unit.unit_cfg.DIG_LICHEN_REMOVED))

            home_plant = PlantManager.get_instance().get_plant(r.home_factory_id)
            l_dist = utils.manhattan_distance(r.loc, home_plant.loc)

            if game_state.real_env_steps < 960 - 2 * l_dist:
                leash = r.calculate_power_leash(game_state)
                # Limit the digs by available power
                n_digs = min(n_digs, (r.power - leash - r.battery//20) // r.dc)

            actions = [r.unit.dig(n=1)] * n_digs
            cost = [r.dc] * n_digs
            return actions, cost


class MissionPlannerSupport(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot, home_plant)
        self.robots = None
        self.digging_robot_uid = None
        self.digging_robot = None
        self.digging_loc = None
        self.res_int = None
        self.supply_in_action = 0

    def is_mission_accomplished(self, game_state):
        """ When a support robot accomplishes/ends the mission, it must reset the actions. """

        self.digging_robot = self.robots.get(self.digging_robot_uid, None) if self.digging_robot_uid else None

        if self.digging_robot_uid and self.digging_robot is None:
            log.debug(f'\t{self.robot.id} something went wrong with the digging robot, aborting')
            return True, None

        elif self.digging_robot and not self.digging_robot.is_chain_needed(game_state):
            log.debug(f'\t{self.robot.id} chain not needed anymore, aborting')
            return True, None

        return False, None

    def has_destination(self, game_state):
        dst_loc = Robot.dst_locs.get(self.robot.id, None)
        if dst_loc and dst_loc in self.digging_robot.missing_links(self.robots):
            return True
        else:
            Robot.dst_locs.pop(self.robot.id, None)
            return False

    def select_destination(self, game_state):
        assert self.robots is not None

        r = self.robot

        # Identify a supply chain, from a digging robot on our primary ice to the nearest factory tile... actually
        # our primary_ice robot may be digging another equally near tile so we need to examine possibly 2 chains

        if not self.digging_robot_uid:
            for q in self.robots.values():
                if ((q.is_primary_ice or q.is_primary_ore) and q.home_factory_id == self.home_plant.id and
                        q.is_chain_needed(game_state)) and \
                        (not q.is_chain_locked(self.robots) or (r.is_heavy() and q.light_links(self.robots))):

                    self.digging_robot_uid = q.id
                    self.digging_robot = q

                    self.res_int = utils.ResourceId.ICE.value if game_state.board.ice[q.loc] else utils.ResourceId.ORE.value
                    break

        if self.digging_robot_uid is None:
            log.debug(f'{r.id} no digging primary digger of {self.home_plant.id} found')
            return None

        unassigned_locs = self.digging_robot.unassigned_links(self.robots)

        if unassigned_locs:
            log.debug(f'{r.id} locations available in the supply path: {unassigned_locs}')
            return unassigned_locs[0]
        elif r.is_heavy():
            log.debug(f'{r.id} the supply chain looks complete, but if we are heavy try to replace light...')
            # Get the robots in the chain
            assigned = self.digging_robot.light_links(self.robots)
            if assigned:
                log.debug(f'{r.id} yes we can replace a light robot at {assigned[0]}...')
                return assigned[0]
        else:
            log.debug(f'{r.id} all spots look assigned, but perhaps robots are not in place..')

        return None

    def activity_at_destination(self, game_state):
        r = self.robot

        # Make sure the chain is complete..
        if not self.digging_robot.is_chain_locked(self.robots):
            log.debug(f'{r.id} the supply chain is incomplete, missing links: {self.digging_robot.missing_links(self.robots)}, '
                      f'waiting for more support')
            return [], None

        # The chain is operational!

        # Whatever cargo it has, transfer "upstream" to the factory and transfer power "downstream" back
        if not r.is_idle() and self.supply_in_action % 20 != 0:
            self.supply_in_action += 1
            return None, None

        # The robot seems to fit in the chain
        self.supply_in_action += 1

        chain = self.digging_robot.chain_path
        dst_loc = Robot.dst_locs[self.robot.id]

        in_chain_ix = chain.index(dst_loc)
        if in_chain_ix == len(chain) - 1:
            # The robot is the last element, so it should be at the factory tile...
            loc_up = dst_loc
        else:
            loc_up = chain[in_chain_ix + 1]

        if in_chain_ix == 0:
            # The robot is next to the digging robot
            loc_down = self.digging_robot.loc
        else:
            loc_down = chain[in_chain_ix - 1]

        d_up = utils.direction_to(dst_loc, loc_up)
        d_down = utils.direction_to(dst_loc, loc_down)

        # Try to maintain the power in robots at 10%
        power_needed = self.digging_robot.get_power_needed_per_step()

        is_factory_end = loc_up == dst_loc
        if is_factory_end:
            # The factory end, must also pick-up power
            discharge = min(2 * power_needed, (r.power - 100) // 20 if r.is_heavy() else 0)
            actions = [
                r.unit.move(0, repeat=0, n=1),
                r.unit.transfer(d_down, utils.ResourceId.POWER.value, 2 * power_needed, repeat=1, n=1),
                r.unit.pickup(utils.ResourceId.POWER.value, 2 * power_needed - discharge, repeat=1, n=1)
            ]
        else:
            if r.is_heavy():
                surplus = np.clip((r.power - (200 + 0.2 * game_state.real_env_steps)) // 20, -2, 2)
            else:
                surplus = 1 if 120 < r.power else (-1 if r.power < 20 else 0)
            discharge = int(2 * power_needed + surplus)
            actions = [
                r.unit.move(0, repeat=0, n=1),
                r.unit.transfer(d_down, utils.ResourceId.POWER.value, discharge, repeat=1, n=1),
                r.unit.transfer(d_up, self.res_int, 75, repeat=1, n=1)
            ]

        return actions, None


class MissionPlannerWait(MissionPlanner):

    def __init__(self, game_state, robot: Robot, home_plant: Plant, steps_to_wait: int, next_mission: Mission):
        super().__init__(game_state, robot, home_plant)
        self.step_0 = game_state.real_env_steps
        self.steps_to_wait = min(10, steps_to_wait)
        self.next_mission = next_mission

    def is_mission_accomplished(self, game_state):
        if game_state.real_env_steps - self.step_0 > self.steps_to_wait:
            log.debug(f'\t{self.robot.id} mission wait accomplished')
            return True, self.next_mission
        else:
            return False, None

    def select_destination(self, game_state):
        if self.home_plant.is_factory_tile(self.robot.loc):
            return self.robot.loc
        else:
            _, _, path = self.robot.find_way(game_state, destination_locs=self.home_plant.get_factory_tiles())
            return path[-1]

    def activity_at_destination(self, game_state):
        if game_state.real_env_steps - self.step_0 == self.steps_to_wait:
            actions = [self.robot.unit.pickup(utils.ResourceId.POWER.value, self.robot.battery - self.robot.power, repeat=0, n=1)]
            return actions, None
        else:
            return None, None


class MissionPlannerSolar(MissionPlanner):

    template = np.array([[1, 2, -9, 2, 1],
                         [2, 3, -9, 3, 2],
                         [-9, -9, -9, -9, -9],
                         [2, 3, -9, 3, 2],
                         [1, 2, -9, 2, 1]])

    def __init__(self, game_state, robot: Robot, home_plant: Plant):
        super().__init__(game_state, robot, home_plant)
        self.step_0 = game_state.real_env_steps
        self.timer = 0

    def is_mission_accomplished(self, game_state):
        if game_state.real_env_steps > 600:
            log.debug(f'\t{self.robot.id} mission SOLAR accomplished')
            return True, None
        else:
            return False, None

    def select_destination(self, game_state):
        # Take a free position, either in a corner of the factory, at its connection tiles or connected to other
        # solars
        score_map = np.clip((9 - board_routines.DistanceMatrix.get_instance(self.robot.loc)) / 5, 0.0, 2.0)
        l = self.home_plant.loc
        floc_slice = slice(max(0, l[0] - 2), min(l[0] + 3, 48)), slice(max(0, l[1] - 2), min(l[1] + 3, 48))
        src_slice = slice(max(0, 2 - l[0]), min(47 - l[0] + 3, 5)), slice(max(0, 2 - l[1]), min(47 - l[1] + 3, 5))
        score_map[floc_slice] += self.template[src_slice]
        # Only corners available

        for uid, dst_loc in Robot.dst_locs.items():
            score_map[dst_loc] = 0

        loc = np.unravel_index(np.argmax(score_map), score_map.shape)
        if score_map[loc] > 0.0:
            log.debug(f'{self.robot.id} destination selected: {loc}')
            self.timer = 0
            return loc
        else:
            log.debug(f'{self.robot.id} could not find a sport to SOLAR, aborting')
            return None

    def activity_at_destination(self, game_state):
        r = self.robot
        T = 50

        if not r.is_idle() and self.timer > 0:
            self.timer -= 1
            return None, None

        else:
            dst_loc = Robot.dst_locs[r.id]
            d_fac = None
            if self.home_plant.is_factory_tile(dst_loc):
                d_fac = 0
            else:
                for t in self.home_plant.get_factory_tiles():
                    if utils.is_adjacent(t, dst_loc):
                        d_fac = utils.direction_to(dst_loc, t)
                        break

            if d_fac is not None:
                charge = min(15, max(0, 6 + (r.power - game_state.real_env_steps) // T))
                actions = [r.unit.move(0, repeat=0, n=1),
                           r.unit.transfer(d_fac, utils.ResourceId.POWER.value, charge, repeat=1, n=1)]
                self.timer = 20
                log.debug(f'{r.id} updating the SOLAR charge')
                return actions, None

            else:
                self.timer = 0
                log.debug(f'{r.id} reseting actions')
                return [], None


