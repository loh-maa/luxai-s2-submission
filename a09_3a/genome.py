import json
from pathlib import Path
import random
import string
import time

import numpy as np

from a09_3a.logger import log
from a09_3a.utils import local_datetime
from a09_3a.utils import to_json


def random_string_wd(n: int) -> str:
    assert n > 0
    return ''.join(random.choice(string.hexdigits.lower()) for _ in range(0, n))


def generate_unique_id(n):
    """ Generate a random n-long hex ID. Make sure it's actually a new one, by trying to insert into the DB. """

    gids = {fp.stem for fp in Path('dev_env').rglob('**/*.json')}

    for n_attempts in range(100):
        gid = random_string_wd(n)
        if gid not in gids:
            return gid

    assert None


genome_default = """{
    "aids": [],
    "created": "2023-02-19 21:26:29 +0700",
    "id": "0000",
    "pid": "unk",
    "placement_weights": {},
    "mission_weights": {}
}"""


class Genome:

    # dpath = 'dev_env/a09_3a/genomes/gen6'

    def __init__(self):
        self.id = '0000'
        self.pid = 'unk'
        self.aids = []
        self.fpath = None
        self.created = local_datetime()

        # All default/neutral values
        self.placement_weights = {
            'ARABLE_EXPOSITION': 1.0,
            'ARABLE_EXPOSITION_DELTA': 1.0,
            'ICE_SRC': 1.0,
            'ICE_ADJACENT': 1.0,
            'ICE_DIV': 1.0,
            'ORE_SRC': 1.0,
            'ORE_DIV': 1.0,
            'FACTORY_SEPARATION': 1.0,
            'BOARD_MIDDLENESS': 1.0,
            'N_CLEAR_CONNECTIONS': 1.0
        }

        # Mission score weights
        self.mission_weights = {
            'home': 1.0,
            'ice': 1.0,
            'ore': 1.0,
            'grazing': 1.0,
            'sabotage': 2.0,
            'hunt': 1.0,
            'recharge': 1.0,
            'guard': 1.0,
            'transfer': 1.0,
        }

        self.flags = {
            'initial_bid': True,
            'probabilistic_mission': False,
            'ramming_enabled': True,
            'evaluate_go_res_old': True
        }

        self.params = {
            'probabilistic_mission_exponent': 3.0
        }

        self.lichen_scores = {}
        self.stats = {}

    @classmethod
    def load(cls, gid=None, fpath=None):
        """ Try to find the right json file, the path may be different depending on how the app was called.
        :param gid: provide either gid and find the .json file recursively
        :param fpath: or provide exact file path
        :return: the loaded genome or None
        """

        if fpath is None:
            assert gid
            fpaths = list(Path().rglob(f'{gid}.json'))
            assert len(fpaths) >= 1
            fpath = fpaths[0]
            assert fpath.exists()

        try:
            with Path(fpath).open(mode='r') as f:
                d = json.load(f)
                genome = Genome()
                genome.fpath = fpath
                genome.deserialize(d)
                return genome
        except FileNotFoundError:
            log.warning(f'Genome file {fpath} not found')
            return None

    @classmethod
    def loads(cls):
        """ Try to find the right json file, the path may be different depending on how the app was called.
        :param gid: provide either gid and find the .json file recursively
        :param fpath: or provide exact file path
        :return: the loaded genome or None
        """
        d = json.loads(genome_default)
        genome = Genome()
        genome.fpath = ''
        genome.deserialize(d)
        return genome

    def randomize(self):
        np.random.seed(int(time.time_ns() & 0xffffffff))
        self.id = generate_unique_id(4)

        for k, v in self.flags.items():
            self.flags[k] = bool(np.random.random() < 0.5)

        self.mutate(gamma=1.0)

    def mutate(self, gamma=0.5):
        self.aids.append(self.id)
        self.pid = self.id
        self.id = generate_unique_id(4)
        log.info(f'Mutating {self.pid} -> {self.id}')

        # Mutate placement weights
        for k, v in self.placement_weights.items():
            if np.random.random() < 0.33:
                self.placement_weights[k] = round(v * 2 ** np.random.uniform(low=-gamma, high=gamma), 2)
        log.info(f'new placement weights: {self.placement_weights}')

        # Mutate mission weights
        for k, v in self.mission_weights.items():
            if np.random.random() < 0.33:
                self.mission_weights[k] = round(v * 2 ** np.random.uniform(low=-gamma, high=gamma), 2)
        log.info(f'new mission weights: {self.mission_weights}')

        for k, v in self.flags.items():
            if np.random.random() < 0.5:
                self.flags[k] = not self.flags[k]
        log.info(f'flags: {self.flags}')

        for k, v in self.params.items():
            if np.random.random() < 0.33:
                self.params[k] = round(v * 2 ** np.random.uniform(low=-gamma, high=gamma), 2)
        log.info(f'params: {self.params}')

        self.lichen_scores = {}

    def serialize(self):
        return {
            'id': self.id,
            'pid': self.pid,
            'aids': self.aids,
            'created': self.created,
            'placement_weights': self.placement_weights,
            'mission_weights': self.mission_weights,
            'flags': self.flags,
            'params': self.params,
            'lichen_scores': self.lichen_scores,
            'lichen_score_mean': self.lichen_score_mean(),
            'lichen_score_combined': self.lichen_score_combined(),
            'stats': self.stats,
        }

    def deserialize(self, d):
        self.id = d['id']
        self.pid = d.get('pid', 'unk')
        self.aids = d.get('aids', [])

        self.created = d.get('created', local_datetime())
        self.placement_weights.update(d.get('placement_weights', {}))
        self.mission_weights.update(d.get('mission_weights', {}))
        self.flags.update(d.get('flags', {}))
        self.params.update(d.get('params', {}))
        self.lichen_scores = d.get('lichen_scores', {})
        # self.lichen_scores_us = d.get('lichen_scores_us', [])
        self.stats = d.get('stats', {})

    def add_score(self, seed, score):
        # If the seed is negative, it means we evaluated as player_1, so negate the score
        self.lichen_scores[str(seed)] = int(score) if seed >= 0 else int(-score)

    def add_stats(self, seed, stats):
        self.stats[seed] = to_json(stats)

    def get_score(self, seed):
        return self.lichen_scores.get(str(seed), None)

    def lichen_score_mean(self):
        return int(sum(self.lichen_scores.values())/len(self.lichen_scores)) if self.lichen_scores else 0

    def lichen_score_combined(self):
        s = sum([sc + 5000 * np.sign(sc) for sc in self.lichen_scores.values()])
        return int(s / (1 + len(self.lichen_scores)))

    def dump_to_file(self, odpath, overwrite=True):
        d = self.serialize()
        odpath = Path(odpath)
        if not odpath.exists():
            odpath.mkdir(exist_ok=True)
        # fpath = self.fpath or odpath / {self.id}.json'
        fpath = odpath / f'{self.id}.json'
        log.info(f'dumping genome {self.id} to {fpath}')
        if overwrite or not fpath.exists():
            with fpath.open(mode='w') as f:
                try:
                    json.dump(to_json(d), f, sort_keys=True, indent=4)
                except TypeError:
                    print(d)
                    exit(1)
        else:
            log.info(f'Cannot dump walker {self.id}, JSON already exists.')
