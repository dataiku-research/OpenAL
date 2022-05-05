from importlib import import_module
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
from pathlib import Path
from collections import defaultdict


class CsvValue:

    def __init__(self, path):
        self.path = path
        try:
            self._data = pd.read_csv(self.path)
            # Make all columns not called value as index
            self._data.set_index(self._data.columns.drop('value').to_list(), inplace=True)
        except (FileNotFoundError, EmptyDataError):
            self._data = None

    def upsert(self, index, value):
        if self._data is None:
            self._data = pd.DataFrame([{**index, 'value': value}])
            self._data.set_index(self._data.columns.drop('value').to_list(), inplace=True)
        else:
            # Check that the index match
            diff = set(index.keys()).difference(set(self._data.index.names))
            if len(diff) != 0:
                raise ValueError('Index mismatch between DB and query: {}'.format(diff))
        
            # Now we just need to update the value if already there otherwise add it
            loc = tuple([index[k] for k in self._data.index.names])
            try:
                self._data.at[loc, 'value'] = value
            except KeyError:
                self._data = self._data.append(pd.DataFrame([[value]], columns=['value'], index=[loc]))
        self._data.to_csv(self.path)

    def get(self, index):
        if self._data is None:
            return None
        index_list = [index[i] for i in self._data.index.names]
        return self._data.loc[tuple(index_list)]


class CsvDb:
    def __init__(self, folder):
        """CSV-based database.
        
        Params:
            folder: Folder where CSV are stored. One CSV per value.
        """

        self.folder = Path(folder)
        self._values = dict()
        if not self.folder.exists():
            self.folder.mkdir()
        else:
            for f in self.folder.iterdir():
                if f.is_dir():
                    continue
                self._values[f.stem] = CsvValue(str(f))
    
    def upsert(self, key, index, value):
        if not key in self._values:
            self._values[key] = CsvValue(str(self.folder / (key + '.csv')))
        self._values[key].upsert(index, value)

    def get(self, key, index):
        if not key in self._values:
            return None
        return self._values[key].get(index)
