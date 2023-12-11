from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy import optimize

from a09_3a import board_routines, utils
from a09_3a import mission_planner
from a09_3a.logger import log
from a09_3a.plant import Plant
from a09_3a.plant_manager import PlantManager
from a09_3a.robot import Robot, Mission


class MissionAllocator:

    def __init__(self):

        self.attack_allocation_mx = None
        self.attack_force_budget = defaultdict(partial(defaultdict, float))
        self.attack_assignment = [] # list of selected candidate entries

    def strategic_matrix(self, game_state, robots):

        our_fids = sorted(game_state.factories[game_state.player].keys())
        opp_fids = sorted(game_state.factories[game_state.opp_player].keys())

        # Distances between factories...
        fcd, lp = PlantManager.get_instance().factory_cross_distance_mx(game_state)
        log.debug(f'smx lichen potentials: {lp}')

        # Our factory want to defend proportionally to its lichen potential and threat,
        # it wants to attack inversely proportionally to its lichen potential and attack efficienty (proximity)
        attack = {}
        defense = {}
        for fid in fcd.keys():
            d = list(fcd[fid].values())
            isolation = (min(d) + sum(d) / len(d)) / 2
            threat = 1.0 - isolation
            defense[fid] = lp[fid]

            pool = [lp[k] * (1.0 - fcd[fid][k])**1.5 for k in fcd[fid].keys()]
            prize = (max(pool) + sum(pool) / len(pool)) / 2
            # We don't take opponent's defenses/units into account
            attack[fid] = (1.0 - lp[fid]) * prize

            log.debug(f'smx: {fid}, lp: {lp[fid]}, isolation: {isolation:.3f}, threat: {threat:.3f}, defense: {defense[fid]:.3f}, pool: {pool}, '
                      f'prize: {prize:.3f}, attack: {attack[fid]:.3f}')

        # Try to figure out which factories should attack which, just try to minimize distances but allocate attacks
        # proportionally to the prize

        prizes = np.array([lp[k] for k in game_state.factories[game_state.opp_player].keys()])
        prizes = prizes / (np.sum(prizes) + 0.001)

        dmx = np.zeros((len(fcd), len(game_state.factories[game_state.opp_player])))
        for i, (fid, f) in enumerate(game_state.factories[game_state.player].items()):
            for j, (fjd, g) in enumerate(game_state.factories[game_state.opp_player].items()):
                dmx[i, j] = utils.manhattan_distance(f.loc, g.loc) / 90

        assert np.all(dmx <= 1.0)

        eff = 1.0 - dmx

        # Figure out how much attack power we want at each plant, fraction of total
        force_plant = {}
        for hfid in our_fids:
            force_plant[hfid] = sum([r.weight * r.power2b() for r in robots.values() if r.home_factory_id == hfid])

        attack_force_plant = {}

        for hfid in our_fids:
            att_def = attack[hfid] + defense[hfid]
            afp = (attack[hfid] + 0.4 * defense[hfid]) / max(0.001, (att_def)) * force_plant[hfid]
            attack_force_plant[hfid] = round(afp, 2)

        attack_force_total = sum(attack_force_plant.values())

        attack_force = np.zeros((1, dmx.shape[0]))
        for hfix, hfid in enumerate(our_fids):
            attack_force[0, hfix] = attack_force_plant[hfid] / max(0.001, attack_force_total)

        def f(a):
            amn = a.reshape(dmx.shape)
            amn /= amn.sum(axis=1).reshape(dmx.shape[0], 1) + 0.00001
            y = attack_force.dot(amn * eff)
            # We want to minimize the difference with the prize target, i.e. maximize diff
            return -np.sum(np.minimum(0.0, y - prizes))

        result = optimize.dual_annealing(f, x0=np.ones((dmx.size, )), bounds=[(0.0, 1.0)]*dmx.size, maxiter=256)
        log.debug(f'smx: result: {result}')

        alloc = result.x.reshape(dmx.shape)
        alloc /= alloc.sum(axis=1).reshape(dmx.shape[0], 1) + 0.00001

        self.attack_allocation_mx = np.round(alloc, 2)
        log.debug(f'smx: attack allocation mx: {self.attack_allocation_mx}')

        self.attack_force_budget = defaultdict(partial(defaultdict, float))
        for hfix, hfid in enumerate(our_fids):
            for tfix, tfid in enumerate(opp_fids):
                afb = attack_force_plant[hfid] * self.attack_allocation_mx[hfix][tfix]
                self.attack_force_budget[hfid][tfid] = round(afb, 2)

        log.debug(f'smx: force plant: {force_plant}')
        log.debug(f'smx: attack force plant: {attack_force_plant}')
        log.debug(f'smx: attack force budget: {self.attack_force_budget}')

    def assign_attack(self, game_state, robots):

        # First, assess the right time to attack is for the robot to arrive at the destination about 20 steps before
        # the end or just to be able to exhaust its power for the travel, digging and maneuvers
        opp_factories = game_state.factories[game_state.opp_player]
        step = game_state.real_env_steps
        steps_left = 1000 - step
        candidates = []
        for uid, r in robots.items():
            # If already assigned, update the budget and continue
            if r.is_mission(Mission.ATTACK):
                force = r.weight * r.power2b()
                hfid = r.home_factory_id
                tfid = r.target_factory_id
                self.attack_force_budget[hfid][tfid] -= force
                log.debug(f'{uid} already has mission attack ({hfid} -> {tfid}, force: {force:.2f})')
                continue

            # Select only from the aggressive robots..
            # todo force stop ice and ore from non-primary units, try to turn grazing to repair
            if not r.is_aggressive():
                continue

            hfid = r.home_factory_id
            targets = [tf for tfid, tf in opp_factories.items() if self.attack_force_budget[hfid][tfid] > 0.0]

            for target_factory in targets:
                _, cost, path = r.find_way(game_state, target_factory.loc)
                # Cut the path so it excludes the opp factory tiles
                path = path[:-3]
                travel_cost = sum(cost[:-3])
                power_aat = (r.power - travel_cost)
                # Power-time available at the target... how much power we can spend in the available time
                ptime_aat = int((steps_left - len(path)) * r.avg_sabotage_cost)
                attstep = int(1000 - power_aat // r.avg_sabotage_cost - len(path))
                # Penalty is when we have more power than time to spend..
                too_late_penalty = max(0, (power_aat - ptime_aat))
                # Stress the cost factor
                score = (power_aat - too_late_penalty - travel_cost) / max(1, power_aat)
                # Prefer strong robots go
                score = round(score + 0.33 * r.power2b(), 3)
                force = round(r.weight * r.power2b(), 3)
                candidates.append((uid, hfid, target_factory.unit_id, score, force, r.weight, attstep, travel_cost))

        # Update current assignments
        self.attack_assignment = []
        mission_planners = {}
        for can in sorted(candidates, key=lambda x: x[3], reverse=True):
            r = robots[can[0]]
            if r.is_mission(Mission.ATTACK):
                # Already assigned
                continue

            log.debug(f'candidate for attack: {can}')

            hfid, tfid = can[1], can[2]
            if self.attack_force_budget[hfid][tfid] > can[4] and abs(can[6] - step) <= 15 and can[3] > 0.2:

                self.attack_assignment.append(can)
                self.attack_force_budget[hfid][tfid] -= can[4]
                log.debug(f'{can[0]} attack assignment: {can}')

                # todo perhaps just schedule for later?
                r.set_mission(Mission.ATTACK)
                r.target_factory_id = can[2]
                mission_planners[r.id] = mission_planner.MissionPlannerAttack(game_state, r, target_fid=can[2],
                                                                              attstep=can[6])

        log.debug(f'unassigned budget: {self.attack_force_budget}')
        return mission_planners

    def balance(self, game_state, mission, home_plant, weight):
        """ Calculate whether the mission is currently under- or over-allocated, for the particular factory and
        given the situation. """

        return 0
