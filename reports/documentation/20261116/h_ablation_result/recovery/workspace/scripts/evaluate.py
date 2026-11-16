"""Approved postprocessing entry; preserve the frozen scientific evaluator."""
import hashlib
import json
import math
from pathlib import Path
import sys


def pinned(binding):
    path = Path(binding['path'])
    if (path.stat().st_size != binding['bytes']
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']):
        raise ValueError('Recovery source/evidence identity differs: '+str(path))
    return path


def serializable_snapshot(value):
    """Only the explicitly unbounded deadline is represented as JSON null."""
    value = dict(value)
    deadline = value['deadline_monotonic']
    if math.isinf(deadline) and deadline > 0:
        value['deadline_monotonic'] = None
        value['auxiliary_limit_disabled'] = True
    elif not math.isfinite(deadline):
        raise ValueError('Invalid deadline; only positive infinity denotes no cap')
    return value


def install_snapshot(budget_type):
    original = budget_type.snapshot

    def snapshot(self):
        return serializable_snapshot(original(self))

    budget_type.snapshot = snapshot


def main():
    if len(sys.argv) < 5 or sys.argv[1] != '--recovery-request':
        raise ValueError('Bound postprocessing recovery request required')
    request_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text())
    if (not request['approved'] or not request['review_closed']
            or request['actions'] != ['confirmation', 'analyze']
            or sys.argv[3] not in request['actions']):
        raise ValueError('This entry cannot train or generate another confirmation batch')
    for b in request['updated_sources']+[request['authorization'], request['review'], request['qualification']]:
        pinned(b)
    if not any(Path(b['path']).resolve() == Path(__file__).resolve() for b in request['updated_sources']):
        raise ValueError('Actual incremental evaluation entry is not bound')
    sys.path.insert(0, str(Path(request['workspace'])/'scripts'))
    import step28_h_ablation_evaluate as scientific
    install_snapshot(scientific.run.Budget)
    sys.argv = [sys.argv[0]]+sys.argv[3:]
    scientific.main()


if __name__ == '__main__':
    main()
