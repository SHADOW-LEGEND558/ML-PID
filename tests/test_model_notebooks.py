"""Verify the distributed training notebooks use the final split and no exports."""
import ast, json, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class ModelNotebookTests(unittest.TestCase):

    def test_training_notebooks_cover_models_without_external_result_writes(self):
        notebooks = [p for p in (ROOT / 'model_testing').glob('*.ipynb') if p.name != 'results_analysis.ipynb']
        self.assertEqual(len(notebooks), 6)
        models = set()
        for path in notebooks:
            book = json.loads(path.read_text(encoding='utf-8'))
            metadata = book['metadata']['mlpid_training']
            models.add(metadata['model'])
            self.assertEqual(metadata['split'], [7000, 1500, 1500])
            self.assertFalse(metadata['external_outputs'])
            for cell in book['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        self.assertNotIn(node.func.attr, {'save', 'savefig', 'to_csv', 'dump', 'write_text', 'write_bytes'})
        self.assertEqual(models, {'linear', 'ridge', 'polynomial_ridge', 'random_forest', 'xgboost', 'standard_mlp'})
