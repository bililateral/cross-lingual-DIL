"""Replay only CAB_s6 stage one through the unchanged frozen training entry.

Preserve the exact calibration fit and pre-calibration state before the old
entry discards them. This entry never proceeds to stage two or confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


class DiagnosticStop(Exception):
    pass


def record_fit(original, directory, scores, labels, **kwargs):
    import numpy as np

    # Forward the original arrays, role and deadline callback without changes.
    np.save(directory / 'calibration_scores.npy', scores, allow_pickle=False)
    fitted = original(scores, labels, **kwargs)
    (directory / 'calibration_fit.json').write_text(
        json.dumps(fitted, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.workspace / 'scripts'))
    import step28_continual_comparison_run as run

    spec = run.m.Spec.from_dict(run.data.read_json(args.spec))
    if spec.run_id != 'CAB_s6_L2R_text_fixed':
        raise ValueError('Only the failed stage-one trajectory is authorized')
    original_fit = run.m.parent.calibration.fit
    original_checkpoint = run.checkpoint
    started = time.monotonic()
    prior = run.data.read_json(args.gate.parent / 'formal/budget.json')
    allowance = 12 * 3600
    if prior['limits']['training'] - prior['used']['training'] < allowance:
        raise ValueError('No room in inherited training bucket')
    identity = dict(spec=spec.identity(), allowance_seconds=allowance,
                    prior_used=prior['used'], prior_failed=prior['failed'],
                    wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scope='One original stage, unchanged fit, no retry/criterion change')
    run.data.write_json(args.output / 'authorization.json', identity)

    def fitted(scores, labels, **kwargs):
        return record_fit(original_fit, args.output, scores, labels, **kwargs)

    def checkpoint(root, method, model, optimizer, stage, calibration, config,
                   first_map, budget):
        if stage.stage != 1:
            raise ValueError('Diagnostic must never enter a later stage')
        optimizer.zero_grad(set_to_none=True)
        metadata = dict(method=method.metadata(), stage=1, adam_step=288,
                        rng=run.m.diag.rng_state(),
                        modes=[v.training for v in model.modules()],
                        first_map=first_map, phase='before_original_checkpoint')
        state = run.save(args.output / 'precalibration.pt', model, optimizer,
                         metadata, budget)
        run.data.write_json(args.output / 'state.json', state)
        (args.output / 'memory.bin').write_bytes(stage.memory.to_bytes())
        original_checkpoint(root, method, model, optimizer, stage, calibration,
                            config, first_map, budget)
        raise DiagnosticStop('Original stage-one checkpoint passed on replay')

    run.m.parent.calibration.fit = fitted
    run.checkpoint = checkpoint
    result = {}
    try:
        run.execute(args.output / 'job', spec, started + allowance, args.gate)
        raise RuntimeError('Stage-one diagnostic unexpectedly returned normally')
    except DiagnosticStop as error:
        result = dict(status='STAGE_ONE_PASS_DIAGNOSTIC_STOP', message=str(error))
    except BaseException as error:
        result = dict(status='DIAGNOSTIC_STOP', error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        result.update(training_seconds=time.monotonic() - started,
                      inherited_training_seconds=prior['used']['training'],
                      formal_restart=False)
        run.data.write_json(args.output / 'result.json', result)


if __name__ == '__main__':
    main()
