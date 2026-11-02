#!/usr/bin/env python3
"""Command-line entry point for predict."""
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
sys.path.insert(0,str(root/'scripts'))
if __name__ == '__main__':
    runpy.run_path(str(root/'src/predict.py'),run_name='__main__')
