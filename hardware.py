"""Small hardware record for the report; no extra Python dependencies."""
import os
import platform
import subprocess
from pathlib import Path


def hardware_info():
    cpu = platform.processor() or platform.machine()
    path = Path('/proc/cpuinfo')
    if path.exists():
        cpu = next((s.split(':', 1)[1].strip() for s in path.read_text().splitlines()
                    if s.startswith('model name')), cpu)
    gpus = []
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True, timeout=10)
        gpus = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    return {'cpu': cpu, 'gpu_names': gpus, 'visible_gpu_selection': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'python': platform.python_version(), 'platform': platform.platform()}
