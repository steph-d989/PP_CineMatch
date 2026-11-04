import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.validation.data_quality import (
    validate_csv_file, validate_json_file, validate_expected_columns,
    validate_file_exists, validate_directory_exists,
)


class DataQualityTests(unittest.TestCase):
    def test_missing_paths(self):
        with TemporaryDirectory() as d:
            p = Path(d) / 'missing.csv'
            self.assertFalse(validate_file_exists(p))
            self.assertTrue(validate_directory_exists(Path(d)))
            self.assertFalse(validate_csv_file(p)['valid'])

    def test_csv_chunks_and_columns(self):
        with TemporaryDirectory() as d:
            p = Path(d) / 'ratings.csv'
            p.write_text('userId,movieId,rating,timestamp\n1,2,4.5,42\n2,3,3,43\n3,4,4,44\n', encoding='utf8')
            r = validate_csv_file(p, chunksize=2)
            self.assertEqual((r['valid'], r['rows'], r['columns']), (True, 3, 4))
            p.write_text('userId,movieId\n1,2\n', encoding='utf8')
            self.assertEqual(validate_csv_file(p)['error'], 'missing_columns')

    def test_csv_empty(self):
        with TemporaryDirectory() as d:
            p = Path(d) / 'movies.csv'
            p.write_text('movieId,title,genres\n', encoding='utf8')
            self.assertEqual(validate_csv_file(p)['error'], 'empty_csv')

    def test_json_and_types(self):
        with TemporaryDirectory() as d:
            p = Path(d) / 'movie.json'
            p.write_text('[]', encoding='utf8')
            self.assertTrue(validate_json_file(p, expected_type=list)['valid'])
            self.assertFalse(validate_json_file(p, expected_type=dict)['valid'])
            p.write_text('{bad json', encoding='utf8')
            self.assertFalse(validate_json_file(p)['valid'])

    def test_expected_columns(self):
        df = pd.DataFrame({'movieId': [1], 'title': ['A']})
        self.assertEqual(validate_expected_columns(df, ['movieId','genres']), {'valid': False, 'missing_columns': ['genres']})


if __name__ == '__main__':
    unittest.main()
