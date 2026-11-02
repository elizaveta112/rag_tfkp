import sys as _sys
from pathlib import Path as _Path
for _directory in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1]/'src', _Path(__file__).resolve().parents[1]/'scripts'):
    _sys.path.insert(0, str(_directory))
"""Driver discovery tests; simulated loader, not a GPU execution test."""
import ctypes
import io
import json
import os
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class CUDASetupTests(unittest.TestCase):
    def run_block(self, path, status=0):
        notebook = Path(__file__).resolve().parents[1]/'notebooks/RAG_TFKP_Qwen35_4B.ipynb'
        source = ''.join(json.loads(notebook.read_text())['cells'][7]['source'])
        block = textwrap.dedent(source[source.index('    # CUDA driver discovery'):source.index("else:\n    env['CMAKE_ARGS']")])
        class Init:
            def __call__(self, flags):
                return status
        class Driver:
            cuInit = Init()
        old_read = Path.read_text
        def read_text(p, *args, **kwargs):
            if str(p) == '/proc/self/maps':
                return '1000-2000 r-xp 0 00:00 0 '+str(path)+'\n'
            return old_read(p, *args, **kwargs)
        env = {'CMAKE_ARGS': '-DGGML_CUDA=ON'}
        with patch.object(ctypes, 'CDLL', return_value=Driver()), \
             patch.object(Path, 'read_text', read_text), \
             patch.object(Path, 'glob', return_value=[]), \
             patch.dict(os.environ), redirect_stdout(io.StringIO()):
            exec(compile(block, str(notebook), 'exec'), {'GPU': True, 'Path': Path, 'os': os, 'env': env, 'ctypes': ctypes})
        return env

    def test_versioned_driver_in_loader_maps_is_found(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mounted-driver/libcuda.so.575.57.08'
            path.parent.mkdir();path.write_bytes(b'SIMULATED DRIVER')
            env = self.run_block(path)
            self.assertIn('-DCUDA_cuda_driver_LIBRARY='+str(path), env['CMAKE_ARGS'])

    def test_stub_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'stubs/libcuda.so.1'
            path.parent.mkdir();path.write_bytes(b'SIMULATED STUB')
            with self.assertRaisesRegex(RuntimeError, 'не найден'):
                self.run_block(path)

    def test_driver_init_failure_stops_gpu_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'libcuda.so.1';path.write_bytes(b'SIMULATED DRIVER')
            with self.assertRaisesRegex(RuntimeError, 'cuInit.*100'):
                self.run_block(path, status=100)


if __name__ == '__main__':
    unittest.main()
