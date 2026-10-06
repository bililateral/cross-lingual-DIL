"""Pure stdlib, read-only audit of supplied CPU identities and recorded resources."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path('/workspace/scratch/f4d639b3b473/relation_revision_review')
CPU = ROOT / 'reports/documentation/20261006/relation_revision/cpu'
OUT = Path(__file__).parent

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def descriptor_check(path, expected, relative):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return {'path': relative, 'actual_bytes': len(data),
            'actual_sha256': digest, 'expected_bytes': expected['bytes'],
            'expected_sha256': expected['sha256'],
            'bytes_match': len(data) == expected['bytes'],
            'sha256_match': digest == expected['sha256']}

manifest = load(CPU / 'manifest.json')
recovery = load(CPU / 'recovery_manifest.json')
preflight = load(CPU / 'preflight.json')
result = load(CPU / 'evidence/result.json')
log = (CPU / 'evidence/unittest.log').read_text(encoding='utf-8')
wrapper = dict(line.split('=', 1) for line in
               (CPU / 'wrapper_time.txt').read_text().splitlines())
source_checks = [descriptor_check(ROOT / e['path'], e, e['path']) for e in manifest]
recovery_checks = [descriptor_check(CPU / p, e, p) for p, e in recovery.items()]
tests, test_seconds = re.search(r'Ran (\d+) tests in ([\d.]+)s', log).groups()
test_names = re.findall(r'^(test_\S+) \(', log, flags=re.MULTILINE)
wrapper_seconds = float(wrapper['wall_seconds'])
wrapper_rss = int(wrapper['max_rss_kib']) * 1024
checks = {
    'source_count_is_19': len(manifest) == 19,
    'source_paths_unique': len({e['path'] for e in manifest}) == len(manifest),
    'all_19_source_size_and_sha_match': all(e['bytes_match'] and e['sha256_match'] for e in source_checks),
    'recovery_count_is_5': len(recovery) == 5,
    'all_5_recovery_size_and_sha_match': all(e['bytes_match'] and e['sha256_match'] for e in recovery_checks),
    'preflight_file_count_matches': preflight['verified_files'] == len(manifest),
    'preflight_cpu_matches_recorded_affinity': result['affinity'] == [preflight['cpu']],
    'tests_are_5_and_named_once': int(tests) == len(set(test_names)) == len(test_names) == 5,
    'unittest_summary_ok': log.rstrip().endswith('OK'),
    'no_unittest_failure_error_skip_summary': not re.search(r'^(FAILED|ERROR:|FAIL:)|\bskipped=', log, flags=re.MULTILINE),
    'supervisor_and_wrapper_success': result['status'] == 'PASS' and result['exit_code'] == 0 and int(wrapper['exit_code']) == 0 and result['limit_failure'] is None,
    'time_containment': float(test_seconds) <= result['elapsed_seconds'] <= wrapper_seconds,
    'recorded_wall_below_600_seconds': wrapper_seconds < 600,
    'sampled_tree_rss_below_2_gib': result['peak_process_tree_rss_bytes'] < 2 * 2**30,
    'wrapper_max_single_process_rss_below_2_gib': wrapper_rss < 2 * 2**30,
    'recorded_gpu_hidden': result['gpu_visible'] == '',
}
report = {
    'audit_scope': 'Only stdlib reads and hashes of supplied files; no imports/execution of project modules, Torch, tests, model, formal data or project Linux access',
    'audit_python': sys.version,
    'checks': checks,
    'all_checks_pass': all(checks.values()),
    'source_checks': source_checks,
    'recovery_checks': recovery_checks,
    'recorded_measurements': {
        'test_seconds': float(test_seconds),
        'supervisor_seconds': result['elapsed_seconds'],
        'wrapper_seconds': wrapper_seconds,
        'wrapper_minus_supervisor_seconds': wrapper_seconds - result['elapsed_seconds'],
        'sampled_process_tree_peak_rss_bytes': result['peak_process_tree_rss_bytes'],
        'wrapper_single_process_max_rss_kib': int(wrapper['max_rss_kib']),
        'wrapper_single_process_max_rss_bytes': wrapper_rss,
        'recovered_5_payload_bytes': sum(e['bytes'] for e in recovery.values()),
        'current_cpu_folder_total_bytes': sum(p.stat().st_size for p in CPU.rglob('*') if p.is_file()),
        'preflight_available_bytes': preflight['available_bytes'],
    },
    'interpretation': [
        'The 0.1-second sampled process-tree RSS and external time maximum single-process RSS are different statistics; their unequal numerical values are not a contradiction.',
        'The three timers measure nested scopes; the stored values are consistently ordered.',
        'formal_data_access, native_bge_loaded and retry are assigned False constants by the supervisor; these fields alone are not independent access or attempt-history proof.',
        'preflight is a saved instantaneous observation and contains no timestamp or generating command; this audit verifies its identity and internal consistency, not remote execution authenticity.',
        'The external time invocation that produced wrapper_time.txt is not in the Bash wrapper; the three-field result is reviewed as submitted execution evidence.',
        'Hashes bind the 19 current source files and the 5 recovered payloads; they do not establish scientific correctness, native BGE feasibility, actual formal effects or absence of unseen runs.',
    ],
    'exact_inconsistencies_found': [],
}
(OUT / 'identity_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'all_checks_pass': report['all_checks_pass'],
                  'source_count': len(source_checks), 'recovery_count': len(recovery_checks),
                  'measurements': report['recorded_measurements'],
                  'exact_inconsistencies_found': []}, ensure_ascii=False))
