import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stress_rejected_plants import recover_rejections


class RejectedLogTests(unittest.TestCase):
    def test_progress_lines_and_literal_reason_dictionary(self):
        text="progress\rDiscarded Plant(zeta=0.7, wn=2.0, tau=0.001, K=1.0): {'resolution_limit': 4}\nprogress\rFailed Plant(zeta=1.0, wn=3.0, tau=0.2, K=0.5): ValueError: example\n"
        records=recover_rejections(text)
        self.assertEqual(len(records),2)
        self.assertEqual(records.iloc[0].original_reason,'resolution_limit')
        self.assertEqual(records.iloc[1].original_reason,'exception')
        self.assertAlmostEqual(records.iloc[0].tau,.001)


if __name__=='__main__':unittest.main()
