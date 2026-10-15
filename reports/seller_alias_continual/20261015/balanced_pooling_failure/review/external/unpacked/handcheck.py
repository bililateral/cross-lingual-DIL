"""Independent review checks in the ChatGPT CPU workspace.
Uses only the unchanged packaged solver and the two declared handwritten inputs.
No repair implementation, optimizer call, model loading, or formal data access.
Usage: python handcheck.py EXTRACTED_INPUT_DIR OUTPUT_JSON
"""
from __future__ import annotations
import hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import mpmath as mp
import torch

def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python handcheck.py EXTRACTED_INPUT_DIR OUTPUT_JSON')
    root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    source = root / 'scripts/step28_balanced_pooling.py'
    expected = '21973808be64245e330cef08d28d841acea3716ba8b07043403a0c22b1958aac'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    sys.path.insert(0, str(root / 'scripts'))
    import step28_balanced_pooling as method
    assert Path(method.__file__).resolve() == source
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    mp.mp.dps = 80
    records = []
    start = time.monotonic()
    for text, attempt in [('1e-8', 'attempt01'), ('1e-4', 'attempt02')]:
        e = float(text)
        p_star = torch.tensor([[1/3-.05-e,.05,e],[.05,1/3-.05-e,e],[e,e,1/3-2*e]], dtype=torch.float64)
        k = torch.log(9*p_star)
        stats = {}
        try:
            method.solve(k[None], stats)
        except method.SolverFailure as exc:
            tb = exc.__traceback__
            while tb and not (tb.tb_frame.f_code.co_name == '_solve' and Path(tb.tb_frame.f_code.co_filename).resolve() == source):
                tb = tb.tb_next
            assert tb is not None
            loc = tb.tb_frame.f_locals
            q = loc['q'][0].clone()
            val = float(loc['mu'][0]+loc['primal'][0])
            residual = exc.details['failed_blocks'][0]
            assert residual['iteration'] == 4096 and stats['failures'] == 1
        else:
            raise AssertionError('Original solver unexpectedly accepted')
        saved = json.loads((root / f'reports/seller_alias_continual/20261015/balanced_pooling_failure/diagnostic_cpu/{attempt}/result.json').read_text())
        qn = q.numpy(); kn = k.numpy()
        em, third, h = mp.mpf(text), mp.mpf(1)/3, mp.mpf(1)/20
        exactp = [[third-h-em,h,em],[h,third-h-em,em],[em,em,third-2*em]]
        qmp = [[mp.mpf(float(qn[i,j])) for j in range(3)] for i in range(3)]
        kl80 = sum(qmp[i][j]*mp.log(qmp[i][j]/exactp[i][j]) for i in range(3) for j in range(3))
        phi80 = sum(qmp[i][j]*(mp.mpf(float(kn[i,j]))-mp.log(9*qmp[i][j])) for i in range(3) for j in range(3))
        construction_delta = max(abs(mp.mpf(float(kn[i,j]))-mp.log(9*exactp[i][j])) for i in range(3) for j in range(3))
        rerr = max(abs(sum(row)-third) for row in qmp)
        cerr = max(abs(sum(qmp[i][j] for i in range(3))-third) for j in range(3))
        total = sum(sum(row) for row in qmp)
        eigens = np.linalg.eigvalsh(9*p_star.numpy().T@p_star.numpy()).tolist()
        records.append(dict(epsilon=text, original_status='ORIGINAL_REJECTED', original_residuals=residual,
            value=val, plan=q.tolist(), value_error_to_analytic_zero=abs(val), plan_l1_error=float((q-p_star).abs().sum()),
            fp32_plan_l1_error=float((q.float().double()-p_star).abs().sum()),
            high_precision_kl_to_exact_p=str(kl80), high_precision_phi_at_actual_float_k=str(phi80),
            high_precision_actual_q_feasible_residual=str(max(rerr,cerr)), high_precision_actual_q_mass_minus_one=str(total-1),
            float_k_construction_max_error=str(construction_delta),
            note_reference='For exact analytic K the optimum is 0. F is 1-Lipschitz in infinity norm, so float K shifts its exact optimum by at most the construction error.',
            local_b_map_eigenvalues=eigens, local_slow_factor=(1-9*e)**2,
            saved_q_max_difference=float(np.abs(qn-np.asarray(saved['final_rounded_plan'])).max()),
            saved_value_difference=abs(val-saved['actual_value']),
            proposed_terminal_output_predicate_only=(residual['feasible_error']<=1e-12 and residual['gap_checked']<=5e-9),
            predicate_note='This is an arithmetic assessment of the saved residuals, not execution or qualification of a repaired solver.'))
    z=method.solve(torch.zeros((1,3,3),dtype=torch.float64))[2]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    result=dict(scope='CHATGPT_ISOLATED_CPU_HANDWRITTEN_CHECK_NOT_PROJECT_LINUX_QUALIFICATION',
        python=platform.python_version(),torch=torch.__version__,source_sha256=expected,
        solver_source_unchanged=True,formal_inputs=False,models_loaded=False,optimizer_calls=0,
        cuda_initialized=torch.cuda.is_initialized(),repair_implemented=False,cases=records,zero_control=z,
        local_elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
