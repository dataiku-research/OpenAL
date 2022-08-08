from bench.csv_db import CsvDb
from tempfile import TemporaryDirectory


def test_dtype():
    with TemporaryDirectory() as tmp_dir_name:
        config = {'a': 'dummy', 'b': '123'}

        csvdb = CsvDb(tmp_dir_name)
        csvdb.upsert('test', config, 2)

        csvdb = CsvDb(tmp_dir_name)
        value = csvdb.get('test', config)

        assert(value.item() == 2)

