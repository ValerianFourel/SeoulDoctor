import tempfile
from pathlib import Path
import unittest

from backend.restore_release import restore, REVISION


class RestoreReleaseTests(unittest.TestCase):
    def test_missing_release_never_writes_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                restore(root / 'missing', root / 'target')
            self.assertFalse((root / 'target' / f'.release-{REVISION}-ready').exists())

    def test_copy_keeps_source_unchanged_and_indexes_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'target'
            for name in ['sources/facilities.parquet', 'sources/reviews.parquet',
                         'chroma_db/chroma.sqlite3', 'search_indexes/active.json']:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            restore(source, target)
            restore(source, target)
            self.assertTrue((target / f'.release-{REVISION}-ready').exists())
            self.assertEqual((target / 'reviews.parquet').read_text(), 'fixture')
            self.assertEqual((target / 'search_indexes/active.json').stat().st_mode & 0o222, 0)
            self.assertTrue((source / 'search_indexes/active.json').stat().st_mode & 0o200)
            (target / 'search_indexes').chmod(0o700)
