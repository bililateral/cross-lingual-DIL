"""Web-scratch counterexample: valid end-state may overflow on begin_stage.

No project Linux, native model, formal data or training. The 48-group statistic
is derived by hand from z=(0,...,0,1), an attainable constant relation head.
Only the supplied actual Memory / Group / Algorithm R code is exercised.
"""
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/scratch/f4d639b3b473/relation_memory_pilot_review/scripts")
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


# Replay the exact pre-repair text-length-176 example against new production code.
report = {"environment": "web scratch Python/NumPy; no PyTorch/BGE/project Linux", "formal_data": False, "training_updates": 0}
memory = make_memory(176)
before = memory.to_bytes()
report["prior_counterexample"] = {"before_bytes": len(before), "maximum_bytes": method.MAXIMUM_BYTES}
try:
    memory.begin_stage(2)
except ValueError as exc:
    report["prior_counterexample"].update(rejected=True, error=str(exc), exact_state_rollback=memory.to_bytes()==before, draw_stage=memory.draw_stage, draw_rng_is_none=memory.draw_rng is None)
else:
    raise AssertionError("Prior counterexample still admits invalid stage")
assert memory.to_bytes()==before and memory.draw_stage==0 and memory.draw_rng is None
# Boundary of live RNG index/count growth, with only independently generated
# handmade public text and padding metadata. No optimizer/update executed.
live = make_memory(2)
live.begin_stage(2)
for _ in range(9):
    live.draw()
live.maps = {"padding": ""}
live.maps["padding"] = "x" * (method.MAXIMUM_BYTES - len(live.to_bytes()))
before = live.to_bytes()
report["live_draw_boundary"] = {"before_bytes": len(before), "draw_count_before": live.draw_count}
try:
    live.draw()
except ValueError as exc:
    report["live_draw_boundary"].update(rejected=True, error=str(exc), exact_state_rollback=live.to_bytes()==before, draw_count_after=live.draw_count)
else:
    raise AssertionError("Exceeded budget draw supplied an input")
assert live.to_bytes()==before and live.draw_count==9
# Non-boundary exact restore and future random stream still behave identically.
normal = make_memory(2)
normal.begin_stage(2)
for _ in range(7):
    normal.draw()
restored = method.Memory.from_bytes(normal.to_bytes())
sequences=[]
for instance in (normal,restored):
    sequence=[]
    for _ in range(281):
        g,z=instance.draw()
        sequence.append(g.uid)
        assert np.array_equal(z,instance.references[g.uid])
    sequences.append(sequence)
assert sequences[0]==sequences[1] and normal.to_bytes()==restored.to_bytes()
report["normal_restore"]={"matching_following_draws":281,"final_draw_count":normal.draw_count,"same_full_bytes":True,"final_bytes":len(normal.to_bytes())}
print(json.dumps(report,ensure_ascii=False,indent=2))
