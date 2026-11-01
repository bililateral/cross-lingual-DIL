"""Exercise the actual measurement callback without importing GPU/model code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import psutil


def callback(path, args):
    tree = ast.parse(path.read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    check = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'check')
    namespace = dict(args=args, time=time, psutil=psutil, shutil=shutil, GIB=2**30)
    exec(compile(ast.Module(body=[check], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['check']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('old', type=Path)
    parser.add_argument('fixed', type=Path)
    args = parser.parse_args()
    context = SimpleNamespace(deadline=time.monotonic()+60, job=args.fixed.parent)
    reserve_bytes = 6*2**30
    try:
        callback(args.old, context)(reserve_bytes)
    except TimeoutError:
        pass
    else:
        raise AssertionError('Original byte/time error was not reproduced')
    check = callback(args.fixed, context)
    check(reserve_bytes)
    try:
        check(shutil.disk_usage(context.job).free)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Actual byte reserve was not enforced')
    context.deadline = time.monotonic()
    try:
        check()
    except TimeoutError:
        pass
    else:
        raise AssertionError('Elapsed-time limit was removed')
    print(json.dumps(dict(status='PASS', old_error_reproduced=True, byte_reserve_accepted=True,
        insufficient_disk_rejected=True, elapsed_time_rejected=True, gpu_calls=0,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (args.old, args.fixed, Path(__file__))})))
