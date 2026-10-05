"""Web-scratch counterexample: valid end-state may overflow on begin_stage.

No project Linux, native model, formal data or training. The 48-group statistic
is derived by hand from z=(0,...,0,1), an attainable constant relation head.
Only the supplied actual Memory / Group / Algorithm R code is exercised.
"""
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/scratch/f4d639b3b473/relation_memory_impl_review/scripts")
import step28_relation_memory as method


controls = [i for i, n in enumerate([3] * 4 + [2] * 8) for _ in range(n)]
labels = tuple(int(controls[i] == controls[j])
               for i, j in itertools.combinations(range(28), 2))


def make_memory(text_length):
    groups = []
    for k in range(48):
        uid = f"hand_group_{k:02}"
        group = method.data.Group(
            uid,
            tuple(f"{uid}_s{i:02}" for i in range(28)),
            tuple(tuple((f"{uid}_r{i:02}_{j}", "物" * text_length, "品" * text_length)
                        for j in range(4)) for i in range(28)),
            labels,
        )
        group.validate()
        groups.append(group)
    memory = method.Memory("ABC")
    memory.reservoir.add_stage(groups)
    memory.h[-1, -1] = 48.0
    memory.b[-1] = -48.0 * 169 / 189
    memory.constant, memory.count, memory.stage = 240.0, 48, 1
    z = np.zeros((32, 378), np.float32)
    z[-1] = 1
    memory.references = {g.uid: z.copy() for g in memory.reservoir.groups}
    memory.reference_keys = {g.uid: memory.group_key(g) for g in memory.reservoir.groups}
    return memory


last = None
for text_length in range(130, 225):
    memory = make_memory(text_length)
    try:
        payload = memory.to_bytes()
    except ValueError:
        break
    last = text_length, memory, payload

text_length, memory, payload = last
assert method.Memory.from_bytes(payload).to_bytes() == payload
none_bytes = len(method.data.json_bytes(None))
memory.begin_stage(2)
rng_bytes = len(method.data.json_bytes(memory.draw_rng.getstate()))
result = {
    "environment": "web scratch; not project Linux/BGE evidence",
    "training_updates": 0,
    "maps": memory.maps,
    "raw_groups": 6,
    "records_per_account": 4,
    "each_text_cjk_characters": text_length,
    "tokenizer_executed": False,
    "count_before": 48,
    "seen_before": 48,
    "end_state_roundtrip_equal": True,
    "serialized_before_bytes": len(payload),
    "budget_bytes": method.MAXIMUM_BYTES,
    "begin_stage_returned": True,
    "draw_rng_json_bytes": rng_bytes,
    "added_rng_bytes": rng_bytes - none_bytes,
    "effective_after_bytes": len(payload) + rng_bytes - none_bytes,
}
try:
    result["after_explicit_check_bytes"] = len(memory.to_bytes())
except ValueError as exc:
    result["after_explicit_check_error"] = str(exc)
    result["draw_still_succeeds_uid"] = memory.draw()[0].uid
    result["draw_count"] = memory.draw_count
print(json.dumps(result, ensure_ascii=False, indent=2))
