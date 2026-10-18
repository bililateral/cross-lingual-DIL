"""Saved-field/source arithmetic only; never imports submitted project code."""
from pathlib import Path
import ast
import hashlib
import json

root = Path('/workspace/scratch/3e46faac4c35/optuna_review_input')
out = Path('/workspace/scratch/3e46faac4c35/optuna_static_review/contributions')
gate = json.loads((root / 'reports/documentation/20261018/optuna_implementation/qualification/cpu_gate.json').read_text())
parts = {'cpu': 300, 'native': 1800, 'search': 36 * 9000,
         'replication': 9 * 9000, 'control': 300, 'evaluation': 2400, 'analysis': 600}
source_paths = ['scripts/step28_optuna_control.py', 'scripts/step28_optuna_search.py',
                'scripts/step28_optuna_run.py', 'scripts/step28_optuna_verify.py',
                'scripts/step28_g_half_run.py', 'scripts/step28_positive_ranking_verify.py',
                'scripts/step28_g_half_verify.py']
identity = {p: {'bytes': (root / p).stat().st_size,
                'sha256': hashlib.sha256((root / p).read_bytes()).hexdigest()}
            for p in source_paths}
run_ast = ast.parse((root / 'scripts/step28_optuna_run.py').read_text())
constructors = [{'line': n.lineno, 'callee': 'Budget'} for n in ast.walk(run_ast)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Budget']
old = [18844.60961239063, 19235.266476024408]
report = {
    'kind': 'STATIC_SOURCE_AND_SAVED_TIMING_ARITHMETIC_ONLY',
    'project_code_imported': False, 'tests_executed': False, 'models_loaded': False,
    'metric_recalculation': False, 'bootstrap': False, 'network_or_project_linux': False,
    'sources': identity, 'campaign_budget_components': parts,
    'campaign_budget_seconds': sum(parts.values()), 'campaign_budget_hours': sum(parts.values()) / 3600,
    'shared_seconds': sum(parts[k] for k in ['control', 'evaluation', 'analysis']),
    'cpu_saved_charged_seconds': gate['cumulative_charged_seconds'],
    'cpu_sum_attempt_charged_seconds': sum(x['charged_seconds'] for x in gate['attempts']),
    'cpu_sum_outer_ns': sum(x['wall_ns'] for x in gate['attempts']),
    'cpu_saved_outer_ns': gate['outer_wall_ns'],
    'cpu_actual_optimizer_calls_saved': gate['actual_cpu_optimizer_calls'],
    'old_projection_values_as_stated_in_budget_amendment': old,
    'old_projection_sum': sum(old), 'proposed_two_run_cap': 18000,
    'old_projection_excess': sum(old) - 18000,
    'old_projection_provenance_limit': 'Values read in budget_amendment.zh.txt; old original native result not independently opened by this subreviewer. No old metric recomputation.',
    'optuna_run_Budget_constructor_calls': constructors, 'mechanical_failures': []}
out.mkdir(exist_ok=True)
(out / 'control_budget.mechanical.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
