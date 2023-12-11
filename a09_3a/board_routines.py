from cachetools import cached, LRUCache
from cachetools.keys import hashkey
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from a09_3a import utils
from a09_3a.logger import log
from a09_3a.path_finding import NhoodMatrix


class DistanceMatrix:

    """ Return a matrix of Manhattan distances from the given location.
    Quick computation based on slicing a precomputed array. """

    instance = None

    @staticmethod
    def initialize(map_size):
        DistanceMatrix.instance = DistanceMatrix(map_size)

    @staticmethod
    def get_instance(loc):
        assert DistanceMatrix.instance is not None
        return DistanceMatrix.instance[loc]

    @staticmethod
    def get_shape():
        """ Get the shape of the matrix/board, can be used by other methods to initialize various maps with the proper
        shape of the board. """
        # Do not call before the matrix is initialized
        assert isinstance(DistanceMatrix.instance, DistanceMatrix)
        return DistanceMatrix.instance.n, DistanceMatrix.instance.n

    def __init__(self, map_size):
        n = map_size
        self.n = n
        x = np.hstack([np.arange(n, 0, -1), np.arange(0, n)])
        y = np.hstack([np.arange(n, 0, -1), np.arange(0, n)])
        xx, yy = np.meshgrid(x, y)
        self.dmx = xx + yy

    def __getitem__(self, loc):
        """ Return a matrix of distances from loc. The location must be within the (n, n) bounds. """
        assert 0 <= loc[0] < self.n and 0 <= loc[1] < self.n
        return self.dmx[self.n - loc[0] : 2 * self.n - loc[0], self.n - loc[1] : 2 * self.n - loc[1]]


def plot_scores(score, lix, labels):
    fig, axs = plt.subplots(3, 3)
    for k in range(min(9, len(lix))):
        axs.flat[k].imshow(score[:, :, lix[k]].T)
        if len(labels) > k:
            axs.flat[k].set_title(labels[k])
    plt.tight_layout()


def simulate_lichen_growth_wild(ldx: np.ndarray, lsx: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Simulate the growth of lichen from all the non-negative cells in lichen_strain.

    :param ldx: lichen_dist board
    :param lsx: lichen_strain board
    :return: both arrays after growth
    """

    cells = [tuple(loc) for loc in np.array(np.nonzero(lsx > -1)).T]
    nhoodmx = NhoodMatrix.get_instance(ldx.shape)

    rubble_value = -1
    wild_treshold = 99
    while cells:
        cell = cells.pop(0)
        for neighbor in nhoodmx.n4[cell]:
            if rubble_value < ldx[cell] + 1 < ldx[neighbor]:
                ldx[neighbor] = ldx[cell] + 1
                lsx[neighbor] = lsx[cell]
                cells.append(neighbor)

            elif wild_treshold < lsx[cell] < lsx[neighbor]:
                # Spread the wild strain with lower ID
                lsx[neighbor] = lsx[cell]
                cells.append(neighbor)

    return ldx, lsx


def compute_heavy_travel_cost_map(from_locs, travel_cost_map: np.ndarray, max_cost: int = 100) -> np.ndarray:
    """ Distance map in terms of power, i.e. through the rubble, starting from given locations reaching as far as
    max_distance. As the result, we have a map where starting cells have value 0 and spreading out up to
    max_distance.

    :param from_locs: can be a single location (tuple) or a Nx2 ndarray
    :param travel_cost_map: travel cost map, including the movement cost for a HEAVY unit
    :param max_cost: compute only as far as cost 100, not for entire map!
    """

    # Our distance map, initially all max values except the starting cells
    xd = np.zeros_like(travel_cost_map) + max_cost

    if isinstance(from_locs, tuple):
        xd[from_locs] = 0
        cells = [from_locs]
    elif isinstance(from_locs, np.ndarray):
        assert from_locs.shape[1] == 2
        xd[tuple(from_locs.T)] = 0
        cells = [tuple(loc) for loc in from_locs]
    else:
        assert False

    nhoodmx = NhoodMatrix.get_instance(xd.shape)

    while cells:
        cell = cells.pop(0)
        for neighbor in nhoodmx.n4[cell]:
            if xd[cell] + travel_cost_map[neighbor] < xd[neighbor]:
                # A shorter path found, spread on
                xd[neighbor] = xd[cell] + travel_cost_map[neighbor]
                cells.append(neighbor)

    return xd


def locate_resources_for_factory(factory_loc, x_res, x_rubble, travel_cost_map, max_d=200, rubble_discount=1.0):
    """ Locate all the ice/ore tiles within max_d power-distance from the factory location. Since the rubble can be
    removed in the long run, discount the power cost due to the rubble as required.

    :return: a list of tuples: [(loc, pdist), (loc, pdist), ...] sorted by pdist, where loc is a location tuple
    """

    factory_tiles = utils.get_factory_tiles(factory_loc)

    # Instead of computing the distance from ice fields to the factory, compute the distance from the
    # factory spreading out..

    # Apply rubble discount
    assert 0.0 <= rubble_discount <= 1.0
    travel_cost_map_rd = np.maximum(20, travel_cost_map - x_res * x_rubble * (1.0 - rubble_discount))

    cost_map = compute_heavy_travel_cost_map(factory_tiles, travel_cost_map_rd, max_cost=max_d)

    # Intersect the distance with the resource tiles
    res_h_distance = x_res * cost_map
    # Replace 0 with max_distance
    res_h_distance[res_h_distance == 0] = max_d
    # Get all ice locations with distances < max_d
    res_loc = list(map(tuple, np.array(np.nonzero(res_h_distance - max_d)).T))
    res_loc_dist = [(loc, res_h_distance[loc]) for loc in res_loc]
    # Sort by distance
    res_loc_dist = sorted(res_loc_dist, key=lambda x: x[1])

    return res_loc_dist


def assign_ice_main_sources(game_state, n_attempts=1):
    best_factory_ice_sources = None
    best_dist = 9999

    for _ in range(n_attempts):
        x_ice = np.copy(game_state.board.ice)
        x_rubble = game_state.board.rubble
        travel_cost_map = np.copy(utils.get_travel_cost_map(game_state, weight=10))

        # Turn off the ice main sources of existing factories, {fid: loc}
        factory_ice_sources = {}
        total_dist = 0
        shuffled_factories = list(game_state.factories[game_state.player].items())
        random.shuffle(shuffled_factories)
        for fid, factory in shuffled_factories:
            # Locate the factory's ice main source
            ice_loc_dist = locate_resources_for_factory(factory.loc, x_ice, x_rubble, travel_cost_map, max_d=300, rubble_discount=0.25)
            if len(ice_loc_dist) == 0:
                log.warning('No source of ice found for an already placed factory...')
                total_dist += 500
            elif len(ice_loc_dist) > 0:
                # At least one source found, add to factory sources and turn off
                factory_ice_sources[fid] = ice_loc_dist[0][0]
                total_dist += ice_loc_dist[0][1]
                x_ice[ice_loc_dist[0][0]] = 0
                travel_cost_map[ice_loc_dist[0][0]] = 200

        if total_dist < best_dist:
            best_dist = total_dist
            best_factory_ice_sources = factory_ice_sources

    return best_factory_ice_sources, best_dist


@cached(cache=LRUCache(maxsize=2), key=lambda game_state, player: (game_state.existing_factories_hash, player))
def get_factory_locs(game_state, player):
    return [f.loc for f in game_state.factories[player].values()]


# Since factories can only vanish, we can use the number of them as the hash
@cached(cache=LRUCache(maxsize=1), key=lambda game_state: game_state.existing_factories_hash)
def get_our_factory_tiles(game_state):
    """ Get all our factories tiles as a list of location tuples. """
    our_ix = utils.get_player_ix(game_state.player)
    our_ft = np.argwhere(game_state.board.factory_map_01 == our_ix)
    our_ft = list(map(tuple, our_ft))
    return our_ft


@cached(cache=LRUCache(maxsize=1), key=lambda game_state: game_state.existing_factories_hash)
def get_opp_factory_tiles(game_state):
    """ Get all the opponent's factories tiles as a list of location tuples. """
    opp_ix = utils.get_player_ix(game_state.opp_player)
    opp_ft = np.argwhere(game_state.board.factory_map_01 == opp_ix)
    opp_ft = list(map(tuple, opp_ft))
    return opp_ft


@cached(cache=LRUCache(maxsize=2), key=lambda game_state, player: (game_state.existing_factories_hash, player))
def get_factory_map(game_state, player):
    """ Get a map with the player's factories and their immediate neighborhoods marked as 1, and all other
    tiles as 0.
    """
    pix = utils.get_player_ix(player)
    fm = np.where(game_state.board.factory_map_01 == pix, 1, 0)
    return np.clip(fm, 0, 1)


@cached(cache=LRUCache(maxsize=2), key=lambda game_state, player: (game_state.existing_factories_hash, player))
def get_factory_neighborhood_map(game_state, player):
    """ Get a map with the player's factories and their immediate neighborhoods marked as 1, and all other
    tiles as 0.
    """
    pix = utils.get_player_ix(player)
    fnm = np.where(game_state.board.factory_map_01 == pix, 1, 0)
    fnm = convolve2d(fnm, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), mode='same')
    return np.clip(fnm, 0, 1)


@cached(cache=LRUCache(maxsize=2), key=lambda game_state, player: (game_state.existing_factories_hash, player))
def get_factories_distance_map(game_state, player):
    """ Get a combined min-distance map from opponents factories. """
    dmx = 100 + np.zeros(shape=DistanceMatrix.get_shape(), dtype=np.int32)
    for f in game_state.factories[player].values():
        for ft in utils.get_factory_tiles(f.loc):
            dmx = np.minimum(dmx, DistanceMatrix.get_instance(ft))
    return dmx


@cached(cache=LRUCache(maxsize=1), key=lambda game_state: game_state.existing_factories_hash)
def get_ice_sources(game_state):
    log.debug('Assign ice main sources to factories..')
    factory_ice_sources, total_dist = assign_ice_main_sources(game_state, n_attempts=10)
    log.debug(f'{factory_ice_sources}, total distance: {total_dist}')
    assert all([isinstance(x, tuple) for x in factory_ice_sources.values()])
    # Return {fid: loc}, however it's not guaranteed all fids are present
    return factory_ice_sources


@cached(cache=LRUCache(maxsize=4), key=lambda game_state, player: hashkey((game_state.existing_factories_hash, player)))
def get_lichen_strains(game_state, player: str):
    team = game_state.teams.get(player, None)
    return team.factory_strains if team is not None else []


@cached(cache=LRUCache(maxsize=8), key=lambda game_state, player: hashkey((game_state.ep_step_hash, player)))
def get_lichen_map(game_state, player: str):
    lichen_map = np.zeros_like(game_state.board.lichen)
    for strain in get_lichen_strains(game_state, player):
        lichen_map = np.where(game_state.board.lichen_strains == strain, game_state.board.lichen, lichen_map)
    return lichen_map


@cached(cache=LRUCache(maxsize=8), key=lambda game_state, player: hashkey((game_state.ep_step_hash, player)))
def get_units_map(game_state, player: str):
    units_map = np.zeros_like(game_state.board.lichen)
    for uid, unit in game_state.units[player].items():
        units_map[unit.loc] = 1 if unit.is_light() else 2
    return units_map


@cached(cache=LRUCache(maxsize=8), key=lambda game_state, player: hashkey((game_state.ep_step_hash, player)))
def get_units_density_map(game_state, player):
    units_map = get_units_map(game_state, player)
    density_map = convolve2d(units_map, utils.kernel_d2, mode='same')
    return density_map


@cached(cache=LRUCache(maxsize=1), key=lambda game_state: game_state.ep_step_hash)
def get_risk_map(game_state):
    opp_units_risk_map = np.zeros_like(game_state.board.rubble, dtype=np.float32)

    m = game_state.env_cfg.map_size
    opp_units = game_state.units[game_state.opp_player]
    for oid, onit in opp_units.items():
        adj_locs = utils.adjacency + onit.loc
        adj_locs = tuple(np.array([al for al in adj_locs if 0 <= al[0] < m and 0 <= al[1] < m]).T)
        opp_units_risk_map[adj_locs] = np.maximum(opp_units_risk_map[adj_locs], onit.get_power_weight())
        # If the unit stays put, its "weight" is lower
        opp_units_risk_map[onit.loc] = np.maximum(opp_units_risk_map[onit.loc], 5 if onit.is_heavy() else 0)

    # Mark opponent factory tiles as power-weight 20
    opp_units_risk_map[tuple(np.array(get_opp_factory_tiles(game_state)).T)] = 20
    # Our own factory tiles have risk 0
    opp_units_risk_map[tuple(np.array(get_our_factory_tiles(game_state)).T)] = 0

    return opp_units_risk_map


@cached(cache=LRUCache(maxsize=1), key=lambda game_state: game_state.existing_factories_hash)
def get_territorial_advantage(game_state):
    """ Positive values are our advantage, negative values are opponent's advantage. +1 is at our factory and when the
    nearest opponent's factory is far away. """
    our_fdm = np.zeros(game_state.board.rubble.shape) + 15
    for f in game_state.factories[game_state.player].values():
        our_fdm = np.minimum(our_fdm, DistanceMatrix.get_instance(f.loc))

    opp_fdm = np.zeros(game_state.board.rubble.shape) + 15
    for f in game_state.factories[game_state.opp_player].values():
        opp_fdm = np.minimum(opp_fdm, DistanceMatrix.get_instance(f.loc))

    tadv = (opp_fdm - our_fdm) / 15
    assert np.all(np.abs(tadv) <= 1.0)
    return tadv


@cached(cache=LRUCache(maxsize=1), key=lambda game_state: hashkey(game_state.episode_id, game_state.real_env_steps//10))
def get_territory_maps(game_state):
    """ Let's say player's territory are tiles which are closer to the player's factory or its lichen, than to the
    opponent's factory or its lichen. No further than 15 though.
    """

    our_lichen_map = get_lichen_map(game_state, game_state.player)
    opp_lichen_map = get_lichen_map(game_state, game_state.opp_player)

    if game_state.real_env_steps >= 500:
        log.debug('check')

    our_ix = utils.get_player_ix(game_state.player)
    our_core_map = np.clip(our_lichen_map, 0, 1)
    our_core_map = np.where(game_state.board.factory_map_01 == our_ix, 1, our_core_map)
    our_core_map = 15 * (1 - our_core_map)
    # Starting tiles are 0, other tiles are 50, spread...

    opp_core_map = get_factory_neighborhood_map(game_state, game_state.opp_player)
    opp_core_map = np.where(opp_lichen_map > 0, 1, opp_core_map)
    opp_core_map = 15 * (1 - opp_core_map)

    ldx = np.minimum(our_core_map, opp_core_map)

    # Our strain == 0, opp strain == 1, other tiles == -1
    strains = np.where(our_core_map == 0, 0, 2)
    strains = np.where(opp_core_map == 0, 1, strains)

    ldx, lsx = simulate_lichen_growth_wild(ldx, strains)

    our_territory_map = np.where(lsx == 0, 1, 0)
    opp_territory_map = np.where(lsx == 1, 1, 0)
    return our_territory_map, opp_territory_map


def lichen_projection(step, n_tiles0, water, arability):
    n_tiles = [n_tiles0]
    for t in range(step + 20, 1000, 20):
        n_tiles.append(round(n_tiles[-1] * (1 + 0.25 * arability) + 12 * arability))
        water -= 20 + n_tiles[-1]
        if water < 0:
            break
    n_tiles_diff = np.diff(np.array(n_tiles), prepend=0)
    lichen = np.sum(np.array(n_tiles_diff) * np.clip(np.arange(10, 20 * len(n_tiles), 20)[::-1], 0, 100))
    return lichen, len(n_tiles)


class BoardManager:
    """ Useful stuff that can be inferred directly from the GameState .. """

    # Singleton pattern
    instance = None

    @staticmethod
    def get_instance():
        assert BoardManager.instance is not None
        return BoardManager.instance

    @staticmethod
    def new_instance(game_state):
        BoardManager.instance = BoardManager(game_state)
        return BoardManager.instance

    def __init__(self, game_state):

        # Pre-calculable constants...
        # Testing a "tuple in np.array" will return True when just a single element is in the array!
        # We want ice_locs as a list of tuples
        self.ice_locs = list(map(tuple, np.argwhere(game_state.board.ice == 1)))
        self.ore_locs = list(map(tuple, np.argwhere(game_state.board.ore == 1)))
        # Risk map exponential averaget difficult to maintain with cache,
        # it has to be updated at every step
        self.risk_map_ema = np.zeros(shape=game_state.board.rubble.shape)

    def new_step(self, game_state):
        self.risk_map_ema = 0.98 * self.risk_map_ema + 0.02 * get_risk_map(game_state)

    def get_risk_map_ema(self):
        return self.risk_map_ema

