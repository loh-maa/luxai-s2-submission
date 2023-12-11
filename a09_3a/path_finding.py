# import cachetools
from cachetools import cached, LRUCache
import cachetools.keys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from a09_3a import utils


class NhoodMatrix:

    instances = {}

    @staticmethod
    def get_instance(shape):
        if shape not in NhoodMatrix.instances:
            NhoodMatrix.instances[shape] = NhoodMatrix(shape)
        return NhoodMatrix.instances[shape]

    def __init__(self, shape):
        self.n4 = np.empty(shape, dtype=np.object)
        # Each element of the array holds a predefined list of neighbors
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.n4[i, j] = [(i + di, j + dj) for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)] if self.is_in_bounds(i + di, j + dj)]

    def __getitem__(self, loc):
        return self.n4[loc]

    def is_in_bounds(self, i, j) -> bool:
        return 0 <= i < self.n4.shape[0] and 0 <= j < self.n4.shape[1]


# We expect this cache to refresh every time a new argument x is used
# the xstr argument is just to provide hashing for the cache (np.ndarrays are non-hashable)
# @cached(cache=LRUCache(maxsize=16))
@cached(cache=LRUCache(maxsize=128), key=lambda x, xstr: cachetools.keys.hashkey(xstr))
def heatmap2graph(x, xstr):
    """ In order to use the Dijkstra shortest path finding algorithm we need to convert a distance-map
    (i.e. the "rubble") to a graph (as a sparse matrix).

    :param x: a distance heatmap, i.e. the rubble + movement cost
    :param xstr: a string representation of the distance heatmap to make the hash (np.ndarrays are not hashable)
    :return: a sparse matrix representing the graph
    """

    M, N = x.shape
    dst_index = np.arange(0, x.size).reshape(x.shape)

    nhoodmx = NhoodMatrix.get_instance(x.shape)

    # How many connections in total?
    assert M > 2 and N > 2
    nc = 4*2 + (2*(M-2) + 2*(N-2)) * 3 + (M-2)*(N-2) * 4

    # Source node index
    row = np.zeros((nc, ), dtype=np.int32)
    # Destination node index
    col = np.zeros((nc, ), dtype=np.int32)
    # Cost
    val = np.zeros((nc, ), dtype=np.int32)

    k = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            src_index = i*N + j
            for neighbor in nhoodmx.n4[(i, j)]:
                # Flat index
                row[k] = src_index
                col[k] = dst_index[neighbor]
                val[k] = x[neighbor]
                k += 1

    coo = coo_matrix((val, (row, col)), dtype=np.int32)
    return coo


def find_shortest_path__ix(coo, from_ix, to_ix, max_d=np.inf):
    """ Return the shortest path from given index to the nearest node in the destination indexes. """

    assert len(to_ix) > 0

    dist_matrix, predecessors = dijkstra(csgraph=coo, directed=True, indices=from_ix, return_predecessors=True, limit=max_d)

    # dist_matrix is len(from_ix) x len(coo)
    mask = np.ones_like(dist_matrix, dtype=bool)
    mask[to_ix] = False
    dist_matrix[mask] = np.max(dist_matrix) + 1
    dst_ix = np.argmin(dist_matrix)

    assert -9999 in predecessors
    path = [dst_ix]

    while True:
        prev = predecessors[path[-1]]
        if prev == -9999:
            break
        path.append(prev)

    return path[::-1]


def find_shortest_path(x, from_loc, to_locs: np.ndarray):
    """ High-level shortest path finding from the loc to any of the destination locs over the distance-map x. """
    assert isinstance(from_loc, tuple) and len(from_loc) == 2 or from_loc.shape == (2, )
    assert isinstance(to_locs, list) or len(to_locs) > 0 and to_locs.shape[1] == 2

    coo = heatmap2graph(x, x.tostring())
    from_ix = np.ravel_multi_index(from_loc, x.shape)
    # to_ix = np.ravel_multi_index(to_locs, x.shape)
    to_ix = np.ravel_multi_index(tuple(np.array(to_locs).T), x.shape)
    assert from_ix not in to_ix

    path = find_shortest_path__ix(coo, from_ix, to_ix)
    path_coords = np.unravel_index(path, shape=x.shape)
    path_coords = np.array(path_coords).T
    assert len(path_coords) >= 2
    # Return as a list of locs
    return list(map(tuple, path_coords))


def find_shortest_path_directions(x: np.ndarray, from_loc: tuple, to_locs: list) -> (list, list, list):
    directions = []
    cost = []
    if utils.is_loc_in(from_loc, to_locs):
        # Technically it should be the same position, actually return empty
        path_coords = [from_loc]
        # directions.append(0)
    else:
        path_coords = find_shortest_path(x, from_loc, to_locs)
        for k in range(1, len(path_coords)):
            directions.append(utils.direction_to(path_coords[k-1], path_coords[k]))
            cost.append(x[tuple(path_coords[k])])
    return directions, cost, path_coords[1:]
