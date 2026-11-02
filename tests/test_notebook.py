"""Notebook commands use the reorganized CLI; publication copy has no outputs."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def notebook(self):
        return json.loads((ROOT/'notebooks/RAG_TFKP_Qwen35_4B.ipynb').read_text())

    def runner(self):
        source=''.join(self.notebook()['cells'][7]['source'])
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run')
        calls=Mock();scope={'subprocess':calls,'PY':'/test/python','ROOT':ROOT}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'notebook_run','exec'),scope)
        return scope['run'],calls

    def test_script_commands_use_dispatcher(self):
        run,calls=self.runner();run('prepare_all.py','--skip-model')
        self.assertEqual(calls.run.call_args.args[0],['/test/python','-u',str(ROOT/'run.py'),'prepare_all.py','--skip-model'])
        self.assertTrue(calls.run.call_args.kwargs['check'])

    def test_python_module_flags_bypass_dispatcher(self):
        run,calls=self.runner();run('-m','unittest','discover','-s','tests','-v')
        self.assertEqual(calls.run.call_args.args[0],['/test/python','-u','-m','unittest','discover','-s','tests','-v'])

    def test_public_notebook_is_clean_and_all_cells_compile(self):
        notebook=self.notebook()
        self.assertEqual(len(notebook['cells']),19)
        for i,c in enumerate(notebook['cells']):
            if c['cell_type']=='code':
                self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])
                compile(''.join(c['source']),f'cell{i}','exec')
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        self.assertNotIn('UPGRADE_COLAB.md',code)
        self.assertNotIn("[PY, 'checkpoint_run.py']",code)
        self.assertNotIn("[PY, '-u', 'prepare_all.py']",code)
        self.assertIn('requirements/core.txt',code)


if __name__=='__main__':unittest.main()
