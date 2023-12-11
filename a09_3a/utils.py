import collections

from cachetools import cached, LRUCache
from cachetools.keys import hashkey
import datetime
from enum import Enum
import time
from typing import Union

import numpy as np


Loc = Union[tuple, np.ndarray]
Locs = Union[list, np.ndarray]


class ResourceId(Enum):
    ICE = 0
    ORE = 1
    WATER = 2
    METAL = 3
    POWER = 4


adjacency = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

kernel_d1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) / 5

kernel_d2 = np.array([[0, 0, 1, 0, 0],
                      [0, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 0, 0]]) / 13


def local_datetime(strfmt='%Y-%m-%d %T %z'):
    # Requires Python 3.6+
    return datetime.datetime.now().astimezone().strftime(strfmt)


def datestamp_string():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')


def to_json(obj):
    """ Convert all numeric types to built-in types (instead of numpy.*), convert all keys to str. """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[str(k)] = to_json(obj[k])
        return out
    else:
        return obj


def is_adjacent(a: Loc, b: Loc):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) == 1


def is_loc_in(loc: Loc, locs: Locs):
    """ Check if the loc is in any of the locs. """
    if isinstance(locs, np.ndarray):
        assert locs.shape[1] == 2
        return any((locs == loc).all(1))
    elif isinstance(locs, list):
        assert isinstance(loc, tuple)
        return loc in locs


@cached(cache=LRUCache(maxsize=4096))
def slice_radius(loc: tuple, r: int, m: int = 48) -> tuple:
    """ Get a slice for the given location and radius. By default, assume the array is square, size 48x48. """
    return slice(max(0, loc[0] - r), min(loc[0] + r + 1, m)), slice(max(0, loc[1] - r), min(loc[1] + r + 1, m))


def is_in_bounds(loc: Loc, map_size: int):
    return 0 <= loc[0] < map_size and 0 <= loc[1] < map_size


@cached(cache=LRUCache(maxsize=128))
def get_factory_tiles(loc: tuple):
    """ Return coordinates of the cells occupied by factory centered at loc.
    :param loc: factory location, as a tuple, list or array
    :return: ndarray 9x2
    """
    i = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    j = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    coords = np.array([i, j]).T + np.array(loc).reshape(1, 2)
    assert coords.shape[1] == 2
    return coords


def manhattan_distance(a: Loc, b: Loc):
    """ Calculate manhattan distances between the location a and b. """
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])


def manhattan_distances(a: Loc, locs: Locs):
    """ Calculate manhattan distances between the location loc and all other locations in locs. """
    return [np.abs(a[0] - l[0]) + np.abs(a[1] - l[1]) for l in locs]


def euclidean_distances(a: Loc, locs: Locs):
    """ Calculate manhattan distances between the location loc and all other locations in locs. """
    return [np.sqrt((a[0] - l[0]) ** 2 + (a[1] - l[1]) ** 2) for l in locs]


def ab_shorter_than_bc(a: Loc, b: Loc, c: Loc):
    """ Location a is closer to b than c to b, i.e. |a-b| < |b-c|. """
    return manhattan_distance(a, b) < manhattan_distance(b, c)


def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = (target[0] - src[0], target[1] - src[1])
    # We operate in IJ coords not JI:
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


def opposite_direction(d):
    return (d + 1) % 4 + 1


dir2str = {0: 'center', 1: 'north', 2: 'east', 3: 'south', 4: 'west'}
cargo2str = {0: 'ice', 1: 'ore', 2: 'water', 3: 'metal', 4: 'power'}


def action2str(aq):
    """ Action queue can be None, int (for a factory) or a list of ndarrays of length 5 (for a unit). """
    if aq is None:
        return None

    if isinstance(aq, int):
        if aq == 0:
            return 'build light'
        if aq == 1:
            return 'build heavy'
        if aq == 2:
            return 'water'
        assert False

    assert isinstance(aq, list) and all([isinstance(a, np.ndarray) and len(a) == 5 for a in aq])

    out = []
    for a in aq:
        if a[0] == 0:
            out.append('move ' + dir2str[a[1]] + (' R' if a[4] else ''))
        elif a[0] == 1:
            out.append('transfer ' + str(a[3]) + ' of ' + cargo2str[a[2]] + ' to ' + dir2str[a[1]] + (' R' if a[4] else ''))
        elif a[0] == 2:
            out.append('pickup ' + str(a[3]) + ' of ' + cargo2str[a[2]] + (' R' if a[4] else ''))
        elif a[0] == 3:
            out.append('dig' + (' R' if a[4] else ''))
        elif a[0] == 4:
            out.append('self-destruct' + (' R' if a[4] else ''))
        elif a[0] == 5:
            out.append('recharge to ' + str(a[3]) + (' R' if a[4] else ''))
    return ', '.join(out)


def dump_stats(game_state, player):
    out = {}

    factories = game_state.factories[player].values()
    # Sum up our lichen
    our_lichen = 0
    for strain in game_state.teams[player].factory_strains:
        our_lichen += np.sum(game_state.board.lichen[game_state.board.lichen_strains == strain])

    out['total'] = {
        'lichen': our_lichen,
        'power': sum([f.power for f in factories]),
        'water': sum([f.cargo.water for f in factories]),
        'metal': sum([f.cargo.metal for f in factories])
    }

    return out


def opp_player(player: str) -> str:
    return "player_1" if player[-1] == '0' else "player_0"


def get_player_ix(player: str) -> int:
    return int(player[-1])


@cached(cache=LRUCache(maxsize=8), key=lambda game_state, weight: hashkey((game_state.ep_step_hash, weight)))
def get_travel_cost_map(game_state, weight: int):
    """ Get the rubble map including the move cost, suitable for robot travel calculation for the current player. """

    # We cannot travel through opponent's factories
    opp_fom_ix = get_player_ix(game_state.opp_player)

    # We use hardcoded weight and move cost values
    if weight == 10:
        travel_cost = game_state.env_cfg.ROBOTS['HEAVY'].MOVE_COST + game_state.board.rubble
    elif weight == 1:
        travel_cost = game_state.env_cfg.ROBOTS['LIGHT'].MOVE_COST + \
                      np.floor(game_state.board.rubble * 0.05).astype(game_state.board.rubble.dtype)
    else:
        assert False, 'Invalid unit weight'

    travel_cost[game_state.board.factory_map_01 == opp_fom_ix] = 1000
    travel_cost.setflags(write=False)
    return travel_cost


@cached(cache=LRUCache(maxsize=4), key=lambda game_state: game_state.ep_step_hash)
def get_rubble_lichen_spread(game_state):
    x_rubble_ls = np.clip(game_state.board.rubble + game_state.board.factory_occupancy_map +
                          1 + game_state.board.ice + game_state.board.ore, 0, 1)
    return x_rubble_ls


def low_pass(x, a, b):
    """ A low-pass envelope. """
    assert a <= b
    if x <= a:
        return 1
    elif x < b:
        return 1 - (x - a) / (b - a)
    else:
        return 0


def hi_pass(x, a, b):
    """ A hi-pass envelope. """
    assert a <= b
    if x <= a:
        return 0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return 1


def game_state_extended(game_state, player: str, opp_player: str, episode_id: int):
    """ Add our custom fields and shortcuts. """

    game_state.player = player
    game_state.opp_player = opp_player
    game_state.episode_id = episode_id

    game_state.solar_gain = [step % game_state.env_cfg.CYCLE_LENGTH < game_state.env_cfg.DAY_LENGTH for step in range(0, 1000)]

    # Turn the factory_occupancy_map to factory_map where factories are marked with the player index instead of the
    # lichen strain
    factory_map_101 = np.zeros_like(game_state.board.factory_occupancy_map, dtype=np.int) - 1
    for player, factories in game_state.factories.items():
        pix = get_player_ix(player)
        for fid, factory in factories.items():
            factory_map_101[factory.pos_slice] = pix
    game_state.board.factory_map_01 = factory_map_101

    # Existing factories hash should include episode ID, because otherwise cache might not refresh properly when
    # playing a series of simulations
    game_state.existing_factories_hash = hash(f'{episode_id}+{sum([len(v) for v in game_state.factories.values()])}')
    game_state.ep_step_hash = hash(f'{episode_id}+{game_state.real_env_steps}')

    return game_state


def modify_cost_by_solar_gain(game_state, step0: int, cost: list, weight: int) -> list:
    """ Modify the cost by the expected solar gain at, assuming the travel starts at step0. """
    cost_mod = [c - game_state.solar_gain[(step0+i) % 1000] * weight for i, c in enumerate(cost)]
    return cost_mod


def flatten_nested_dict(d, parent_key='', sep='.', prefix_w_parent=False):
    """ Convert a nested dict to a flat dict. """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key and prefix_w_parent else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dict(v, parent_key=new_key, sep=sep, prefix_w_parent=prefix_w_parent).items())
        else:
            items.append((new_key, v))
    d_flat = dict(items)
    assert len(d_flat) == len(items)
    return d_flat


def compare_stats(p, q):
    pf = flatten_nested_dict(p, prefix_w_parent=True)
    qf = flatten_nested_dict(q, prefix_w_parent=True)

    out = []
    for k, v in pf.items():
        out.append(f'{k}: {v}, {qf.get(k, "na")}')
    return '\n'.join(out)
