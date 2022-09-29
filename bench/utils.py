import sys
from enum import Enum
import io
from.csv_db import CsvDb
from pathlib import Path
from cardinal.uncertainty import MarginSampler, ConfidenceSampler, EntropySampler
from cardinal.random import RandomSampler
from cardinal.clustering import KCentroidSampler, MiniBatchKMeansSampler, KCenterGreedy
from bench.samplers import TwoStepIncrementalMiniBatchKMeansSampler, TwoStepMiniBatchKMeansSampler, AutoEmbedder
from sklearn.cluster import MiniBatchKMeans


REFERENCE_METHODS = {
    'random': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=params['seed']),
    'margin': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
    'confidence': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
    'entropy': lambda params: EntropySampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
    'kmeans': lambda params: KCentroidSampler(MiniBatchKMeans(n_clusters=params['batch_size'], n_init=1, random_state=params['seed']), batch_size=params['batch_size']),
    'wkmeans': lambda params: TwoStepMiniBatchKMeansSampler(10, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
    'iwkmeans': lambda params: TwoStepIncrementalMiniBatchKMeansSampler(10, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
    'kcenter': lambda params: KCenterGreedy(AutoEmbedder(params['clf'], X=params['train_dataset']), batch_size=params['batch_size']),
}

def is_reference(method):
    return method in REFERENCE_METHODS

def get_method_db(experimental_parameters, method):
    experiment_name = experimental_parameters['experiment_name']
    experiment_folder = Path('./experiment_results') / experiment_name
    if is_reference(method):
        db = CsvDb(experiment_folder / 'db')
    else:
        db = CsvDb(experiment_folder / 'user_{}'.format(method) / 'db')
    return db


class Tee(io.StringIO):
    class Source(Enum):
        STDOUT = 1
        STDERR = 2

    def __init__(self, clone=Source.STDOUT, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._clone = clone

        if clone == Tee.Source.STDOUT:
            self._out = sys.stdout
        elif clone == Tee.Source.STDERR:
            self._out = sys.stderr
        else:
            raise ValueError("Clone has to be STDOUT or STDERR.")

    def write(self, *args, **kwargs):
        self._out.write(*args, **kwargs)
        return super().write(*args, **kwargs)

    def __enter__(self):
        if self._clone == Tee.Source.STDOUT:
            sys.stdout = self
        else:
            sys.stderr = self
        self.seek(io.SEEK_END)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._clone == Tee.Source.STDOUT:
            sys.stdout = self._out
        else:
            sys.stderr = self._out
        self.seek(0)
        return False