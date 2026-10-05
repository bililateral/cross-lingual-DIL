"""Targeted production Memory byte measurements; no torch and no training."""
from __future__ import annotations
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'snapshot/scripts'))
import numpy as np
import step28_relation_memory as method
started=time.monotonic()
original_source=Path('/workspace/scratch/f4d639b3b473/relation_memory_impl_review/scripts/step28_relation_memory.py')
sha_before=hashlib.sha256(original_source.read_bytes()).hexdigest()
limit=1048576
controls=[i for i,n in enumerate([3]*4+[2]*8) for _ in range(n)]
labels=tuple(int(controls[i]==controls[j]) for i,j in itertools.combinations(range(28),2))

def handmade_native_style(uid, records=8, text_length=254):
    def text(i,j,channel):
        prefix=f'商款{chr(0x4e00+i)}{chr(0x4e40+j)}{chr(0x4e60+channel)}'
        return prefix+'物'*(text_length-len(prefix))
    group=method.data.Group(uid,tuple(f'{uid}_s{i:02}' for i in range(28)),
        tuple(tuple((f'{uid}_i{i:02}_{j}',text(i,j,0),text(i,j,1)) for j in range(records))
                    for i in range(28)),labels)
    group.validate()
    assert all(len(t)==text_length and len(d)==text_length for rows in group.items for _,t,d in rows)
    return group

def fixture(text_length):
    memory=method.Memory('ABC')
    memory.stage=1;memory.count=48;memory.constant=240.;memory.reservoir.seen=48
    memory.h=np.eye(32,dtype=np.float64)*48
    memory.b=np.zeros(32,dtype=np.float64)
    memory.reservoir.groups=[handmade_native_style(f'budget_g{i:02}',text_length=text_length) for i in range(6)]
    for i,g in enumerate(memory.reservoir.groups):
        z=np.random.default_rng(i+12).uniform(-.8,.8,(32,378)).astype(np.float32);z[-1]=1
        memory.references[g.uid]=z;memory.reference_keys[g.uid]=memory.group_key(g)
    return memory

rows=[]
for text_length in (5,254):
    memory=fixture(text_length)
    row={'text_characters_each_channel':text_length,'group_count':6,'accounts_per_group':28,
         'records_per_account':8,'channels_per_record':2,'stage':1,'count':48,'reservoir_seen':48,
         'raw_CJK_UTF8_text_bytes':6*28*8*2*text_length*3}
    try:
        payload=memory.to_bytes()
        row.update(original_1MiB_status='ACCEPTED',complete_payload_bytes=len(payload))
    except ValueError as exc:
        row.update(original_1MiB_status='REJECTED',error=str(exc))
        # Measurement-only increase of BOTH in-memory allowances. No source file edit.
        original_cap=memory.reservoir.maximum_bytes
        old_module_cap=method.MAXIMUM_BYTES
        memory.reservoir.maximum_bytes=8*2**20;method.MAXIMUM_BYTES=8*2**20
        payload=memory.to_bytes()
        row.update(measurement_only_cap=8*2**20,complete_payload_bytes=len(payload),
                   raw_reservoir_json_bytes=len(memory.reservoir.to_bytes()),
                   measurement_changes_metadata_length=False)
        assert len(str(original_cap))==len(str(memory.reservoir.maximum_bytes))
        memory.reservoir.maximum_bytes=original_cap;method.MAXIMUM_BYTES=old_module_cap
        try: memory.to_bytes()
        except ValueError as exc: row['reverted_1MiB_status']='REJECTED: '+str(exc)
        else: raise AssertionError('Expected original budget to reject')
    rows.append(row)

# Legitimately serializable fixed-size metadata places a small-text fixture just below budget.
memory=fixture(5)
memory.maps={'fixed_metadata_padding':''}
base_len=len(memory.to_bytes())
memory.maps['fixed_metadata_padding']='x'*(limit-base_len-64)
before=memory.to_bytes()
assert len(before)==limit-64
memory.begin_stage(2)
boundary={'stage1_payload_bytes':len(before),'spare_bytes':limit-len(before),
          'begin_stage_returned_normally':True,'draw_stage':memory.draw_stage}
try: memory.to_bytes()
except ValueError as exc: boundary['serialization_after_begin_stage']='REJECTED: '+str(exc)
else: raise AssertionError('Expected RNG state to exceed remaining 64 bytes')
old_module_cap=method.MAXIMUM_BYTES;method.MAXIMUM_BYTES=2*2**20
actual_after=len(memory.to_bytes())
method.MAXIMUM_BYTES=old_module_cap
group,_=memory.draw()
boundary.update(stage2_complete_payload_bytes=actual_after,
    new_rng_header_growth_bytes=actual_after-len(before),over_budget_bytes=actual_after-limit,
    draw_after_oversized_begin_still_succeeds=True,drawn_uid=group.uid,draw_count=memory.draw_count,
    scope='Production begin_stage/draw paths; count=1 native fixture does not call these.')

# Stronger normal-shape boundary: raw text alone, no filler metadata.
raw_memory=fixture(86)
raw_initial=len(raw_memory.to_bytes())
target_bytes=limit-1000
remaining_chars=(target_bytes-raw_initial)//3
assert remaining_chars>0
for index,g in enumerate(raw_memory.reservoir.groups):
    accounts=[]
    for rows_in_account in g.items:
        rows_out=[]
        for uid,title,description in rows_in_account:
            amount=min(254-len(title),remaining_chars)
            title += '物'*amount; remaining_chars-=amount
            rows_out.append((uid,title,description))
        accounts.append(tuple(rows_out))
    changed=method.data.Group(g.uid,g.sellers,tuple(accounts),g.labels)
    changed.validate()
    raw_memory.reservoir.groups[index]=changed
    raw_memory.reference_keys[changed.uid]=raw_memory.group_key(changed)
assert remaining_chars==0
all_lengths=[len(text) for g in raw_memory.reservoir.groups for rows in g.items
             for _,title,desc in rows for text in (title,desc)]
raw_before=len(raw_memory.to_bytes())
assert 0<limit-raw_before<=1002
assert min(all_lengths)>=2 and max(all_lengths)<=254
assert raw_memory.maps=={}
raw_memory.begin_stage(2)
raw_boundary={'maps':{},'groups_validate':True,'records_per_account':8,
    'minimum_field_characters':min(all_lengths),'maximum_field_characters':max(all_lengths),
    'stage1_complete_payload_bytes':raw_before,'spare_bytes':limit-raw_before,
    'begin_stage_returned_normally':True}
try:raw_memory.to_bytes()
except ValueError as exc:raw_boundary['serialization_after_begin_stage']='REJECTED: '+str(exc)
else:raise AssertionError('Expected raw-text near-limit memory to exceed budget after begin_stage')
old_module_cap=method.MAXIMUM_BYTES;method.MAXIMUM_BYTES=2*2**20
raw_after=len(raw_memory.to_bytes())
method.MAXIMUM_BYTES=old_module_cap
raw_g,_=raw_memory.draw()
raw_boundary.update(stage2_complete_payload_bytes=raw_after,
    rng_header_growth_bytes=raw_after-raw_before,over_budget_bytes=raw_after-limit,
    draw_after_oversized_begin_still_succeeds=True,drawn_uid=raw_g.uid,draw_count=raw_memory.draw_count,
    note='All Group validations pass; CJK fields <=254 characters, but no tokenizer/native claim.')

report={'environment':{'python':sys.version,'numpy':np.__version__,'torch_loaded':'torch' in sys.modules,
           'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')},
    'scope':'Web production non-torch functions; synthetic state; not trained 48 groups, not tokenized by BGE.',
    'maximum_and_short_text_measurement':rows,'begin_stage_budget_boundary':boundary,
    'raw_text_only_begin_stage_budget_boundary':raw_boundary,
    'source_unchanged':hashlib.sha256(original_source.read_bytes()).hexdigest()==sha_before,
    'elapsed_seconds':time.monotonic()-started}
(root/'extra_budget_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
