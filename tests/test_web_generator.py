"""Keep browser tuning synchronized with the Python data generator."""
import hashlib
import json
import unittest
from pathlib import Path


class WebGeneratorTests(unittest.TestCase):
    def test_browser_snapshot_matches_current_generator(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'src' / 'data_gen.py').read_text(encoding='utf-8')
        exported = (root / 'docs' / 'generator-source.js').read_text(encoding='utf-8')
        payload = json.loads(exported.split('window.PID_GENERATOR = ', 1)[1].rstrip(';\n'))
        self.assertEqual(payload['sha256'], hashlib.sha256(source.encode()).hexdigest(),
                         'Run python scripts/export_web_generator.py after changing the generator.')
        compile(payload['source'], '<browser generator>', 'exec')


if __name__ == '__main__':
    unittest.main()
