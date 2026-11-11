#!/usr/bin/env python3
"""Fail-closed admission of the optional CPU tower before image-enabled startup."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys


def checked_file(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('vision input requires a SHA256 pin')
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ValueError('vision input must be a regular file')
    with p.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError('vision input SHA256 mismatch')


def verify(bias, bias_sha, python='', weights='', weights_sha=''):
    checked_file(bias, bias_sha)
    if python or weights or weights_sha:
        if not all((python, weights, weights_sha)):
            raise ValueError('vision encoder needs Python, weights and their SHA256 together')
        checked_file(weights, weights_sha)
        if not Path(python).is_absolute() or not Path(python).is_file():
            raise ValueError('vision Python must be an absolute executable path')
        subprocess.run([python, '-I', '-c',
            'import torch,numpy,PIL;assert torch.__version__=="2.9.0+cpu";'
            'assert numpy.__version__=="2.3.5";assert PIL.__version__=="12.3.0"'],
            check=True, timeout=30, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--bias', required=True)
        parser.add_argument('--bias-sha256', required=True)
        parser.add_argument('--python', default='')
        parser.add_argument('--weights', default='')
        parser.add_argument('--weights-sha256', default='')
        args = parser.parse_args()
        verify(args.bias, args.bias_sha256, args.python, args.weights, args.weights_sha256)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        sys.exit('vision preflight: ' + str(exc))
