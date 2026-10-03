from pathlib import Path
import difflib
import hashlib
import json

home = Path(__file__).resolve().parent
old = home / 'independent_er_source.py'
new = home / 'independent_er_source_v2.py'
text = old.read_text()
before = '''            paired_records = [read_json(OLD / "run/updates" / f"{order}_{arm}_stage{stage}.json")
                              for arm in ("seq", "er")]
            paired_records += [read_json(WEIGHT / "run/updates" / f"{order}_{arm}_stage{stage}.json")
                               for arm in ("half", "quarter")]
            for j, paired in enumerate(paired_records):
                assert paired["current_ids"] == schedule and paired["current_dropout_stream"] == stream
                if j:
                    assert paired["history_ids"] == history
'''
after = '''            # The delivered historical update logs contain original ER only.
            # Never present absent SEQ/quarter logs as independently inspected.
            paired = read_json(OLD / "run/updates" / f"{order}_er_stage{stage}.json")
            assert paired["current_ids"] == schedule and paired["current_dropout_stream"] == stream
            assert paired["history_ids"] == history
'''
assert text.count(before) == 1
revised = text.replace(before, after)
assert not new.exists()
new.write_text(revised)
diff = ''.join(difflib.unified_diff(text.splitlines(True), revised.splitlines(True), fromfile=old.name, tofile=new.name))
(home / 'revision_v1_to_v2.diff').write_text(diff)
note = {'reference_failure': 'ER-REF-01', 'original_run': 'runs/er_source_independent_v1',
        'cause': 'Independent reference assumed unprovided historical SEQ and half/quarter update JSON files existed.',
        'fix': 'Use independently regenerated frozen schedule and supplied original ER six update records. Explicitly exclude absent historical per-step logs from direct coverage.',
        'input_project_changed': False,
        'before': {'path': old.name, 'sha256': hashlib.sha256(old.read_bytes()).hexdigest()},
        'after': {'path': new.name, 'sha256': hashlib.sha256(new.read_bytes()).hexdigest()}}
(home / 'revision_v1_to_v2.json').write_text(json.dumps(note, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(note, ensure_ascii=False, indent=2))
