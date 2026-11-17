"""Git text normalization must not break or bypass study integrity checks."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from provenance import matches_recorded_hash


class ProvenanceTests(unittest.TestCase):
    def test_text_checks_accept_only_line_ending_conversion(self):
        lf = b'K,wn\n1.0,2.0\n'
        crlf = lf.replace(b'\n', b'\r\n')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'plants.csv'
            for current, recorded in [(lf, lf), (lf, crlf), (crlf, lf)]:
                path.write_bytes(current)
                self.assertTrue(matches_recorded_hash(path, hashlib.sha256(recorded).hexdigest()))
            path.write_bytes(lf.replace(b'2.0', b'3.0'))
            self.assertFalse(matches_recorded_hash(path, hashlib.sha256(crlf).hexdigest()))

    def test_binary_checks_remain_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.gz'
            path.write_bytes(b'weights\r\n')
            self.assertFalse(matches_recorded_hash(path, hashlib.sha256(b'weights\n').hexdigest()))
