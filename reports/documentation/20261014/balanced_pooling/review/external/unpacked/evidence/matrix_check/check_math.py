"""Independent tiny matrix reference. Python standard library only.
No import or execution of uploaded project code. No scientific dataset or weights.
Exact 2x2 formula and independent scalar bisection for 2xm supplement Sinkhorn.
Finite-iteration violations are deliberately measured, NOT hidden or fixed.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

Matrix = list[list[float]]
def flat(a: Matrix) -> list[float]: return [v for r in a for v in r]
def transpose(a: Matrix) -> Matrix: return [list(r) for r in zip(*a)]
def mean(a: Matrix) -> float: return math.fsum(flat(a))/len(flat(a))
def lse(x: list[float]) -> float:
    m = max(x)
    return m + math.log(math.fsum(math.exp(v-m) for v in x))
def sigmoid(x: float) -> float:
    if x >= 0: return 1/(1+math.exp(-x))
    y = math.exp(x)
    return y/(1+y)
def logcosh(x: float) -> float:
    return abs(x)+math.log1p(math.exp(-2*abs(x)))-math.log(2)
def diff(a: Matrix, b: Matrix) -> float:
    return max(abs(x-y) for x,y in zip(flat(a),flat(b)))
def objective(k: Matrix, p: Matrix, tau: float) -> float:
    nm = len(k)*len(k[0])
    return math.fsum(p[i][j]*(k[i][j]-tau*math.log(p[i][j]*nm))
                     for i in range(len(k)) for j in range(len(k[0])) if p[i][j]>0)
def reference_2xm(k: Matrix, tau: float) -> tuple[float,Matrix,str]:
    if len(k)!=2: raise ValueError('Reference supports two rows only')
    m=len(k[0])
    if m==2:
        mu=mean(k); d=(k[0][0]-k[0][1]-k[1][0]+k[1][1])/4
        p=sigmoid(2*d/tau)
        return mu+tau*logcosh(d/tau), [[p/2,(1-p)/2],[(1-p)/2,p/2]], 'exact_2x2'
    ds=[(k[0][j]-k[1][j])/tau for j in range(m)]
    lo=min(ds)-2; hi=max(ds)+2
    # Only one dual difference is unknown after imposing column masses 1/m.
    for _ in range(180):
        c=(lo+hi)/2
        if math.fsum(sigmoid(v-c) for v in ds)/m > .5: lo=c
        else: hi=c
    c=(lo+hi)/2
    r=[sigmoid(v-c)/m for v in ds]
    p=[r,[1/m-x for x in r]]
    return objective(k,p,tau),p,'independent_scalar_bisection_2xm'
def sinkhorn(k: Matrix, tau: float, steps: int, tol: float | None) -> dict:
    n,m=len(k),len(k[0]); a=1/n;b=1/m
    if tau<=0 or any(len(r)!=m for r in k) or not all(map(math.isfinite,flat(k))):
        raise ValueError('Invalid manual matrix')
    v=[0.0]*m; u=[0.0]*n; used=0; residual=math.inf
    for used in range(1,steps+1):
        u=[-math.log(n)-lse([k[i][j]/tau+v[j] for j in range(m)]) for i in range(n)]
        v=[-math.log(m)-lse([k[i][j]/tau+u[i] for i in range(n)]) for j in range(m)]
        lp=[[k[i][j]/tau+u[i]+v[j] for j in range(m)] for i in range(n)]
        p=[[math.exp(x) for x in r] for r in lp]
        residual=max([abs(math.fsum(r)-a) for r in p]+
                     [abs(math.fsum(p[i][j] for i in range(n))-b) for j in range(m)])
        if tol is not None and residual<=tol: break
    val=math.fsum(p[i][j]*(k[i][j]-tau*(lp[i][j]+math.log(n*m)))
                  for i in range(n) for j in range(m))
    return {'value':val,'P':p,'iterations':used,'max_marginal_error':residual,
            'converged_to_requested_tolerance':None if tol is None else residual<=tol}
def derivative(fn, k: Matrix, eps: float) -> Matrix:
    out=[]
    for i in range(len(k)):
        r=[]
        for j in range(len(k[0])):
            plus=[row[:] for row in k]; minus=[row[:] for row in k]
            plus[i][j]+=eps;minus[i][j]-=eps
            r.append((fn(plus)-fn(minus))/(2*eps))
        out.append(r)
    return out

def main() -> None:
    d=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out=Path(sys.argv[2]);tau=d['tau'];eps=d['finite_difference_step']
    cases=d['cases'];tol=d['solver_tolerance'];limit=d['solver_max_iterations']
    checks=[]; results={}; max_reference_error=0.;max_gradient_error=0.
    def check(name: str, ok: bool, **details) -> None:
        checks.append({'name':name,'passed':bool(ok),**details})
        print(('PASS ' if ok else 'FAIL ')+name+' '+json.dumps(details,ensure_ascii=False))
    def solve(k):return sinkhorn(k,tau,limit,tol)
    for name,k in cases.items():
        if name=='slow22':continue  # Deliberately tested only at truncated iterations below.
        r=solve(k);f=r['value'];mu=mean(k);g=tau*(lse([v/tau for v in flat(k)])-math.log(len(flat(k))))
        r.update(mean=mu,G=g,max=max(flat(k)))
        check(name+'_convergence',r['converged_to_requested_tolerance'],residual=r['max_marginal_error'],iterations=r['iterations'])
        check(name+'_bounds',mu-1e-11<=f<=g+1e-11 and g<=max(flat(k))+1e-11)
        if len(k)==2:
            rf,rp,method=reference_2xm(k,tau);err=abs(f-rf)
            max_reference_error=max(max_reference_error,err)
            r.update(reference_value=rf,reference_P=rp,reference_method=method,value_error=err,P_error=diff(r['P'],rp))
            check(name+'_independent_reference',err<1e-10 and diff(r['P'],rp)<1e-10,value_error=err,P_error=diff(r['P'],rp))
        else:
            rf=tau*math.log((math.exp(2/tau)+7)/8)
            check(name+'_symmetry_closed_form',abs(f-rf)<1e-11,error=abs(f-rf))
        results[name]=r
    check('cover_and_row_distinguished',results['cover22']['value']>results['row22']['value'])
    check('cover_and_row_G_equal',abs(results['cover22']['G']-results['row22']['G'])<1e-14)
    check('isolated_and_cover_F_equal',abs(results['isolated22']['value']-results['cover22']['value'])<1e-11)
    check('opposite_centered_interaction_F_equal',abs(results['anti_cover22']['value']-results['cover22']['value'])<1e-11)
    c=1+.5*logcosh(1); positive=[[c,c],[c,c]];pf=solve(positive)['value']
    counter={'positive_constant_K':positive,'positive_mean':c,'negative_mean':1.,
             'positive_F':pf,'negative_F':results['cover22']['value']}
    check('proposal_ranking_counterexample',c>1 and pf<results['cover22']['value'],**counter)
    k=cases[d['property_base']];n,m=len(k),len(k[0]);mu=mean(k)
    rows=[math.fsum(r)/m for r in k]; cols=[math.fsum(k[i][j] for i in range(n))/n for j in range(m)]
    w=[[k[i][j]-rows[i]-cols[j]+mu for j in range(m)] for i in range(n)]
    f=solve(k)['value'];fw=solve(w)['value']
    shift=[[k[i][j]+d['row_shift'][i]+d['column_shift'][j] for j in range(m)] for i in range(n)]
    sf=solve(shift)['value'];expected=f+math.fsum(d['row_shift'])/n+math.fsum(d['column_shift'])/m
    perm=[[k[i][j] for j in d['column_permutation']] for i in d['row_permutation']]
    props={'W':w,'F_K':f,'mean':mu,'F_W':fw,'decomposition_error':abs(f-mu-fw),
           'shifted_K':shift,'shift_F':sf,'shift_expected':expected,'shift_error':abs(sf-expected),
           'transpose_error':abs(f-solve(transpose(k))['value']),
           'permutation_error':abs(f-solve(perm)['value'])}
    for key in ['decomposition_error','shift_error','transpose_error','permutation_error']:
        check(key,props[key]<1e-10,error=props[key])
    gradients={}
    for name in ['cover22','rect23','rect28']:
        k=cases[name];rf,rp,_=reference_2xm(k,tau)
        fd=derivative(lambda x:reference_2xm(x,tau)[0],k,eps)
        err=diff(fd,rp);max_gradient_error=max(max_gradient_error,err)
        gradients[name]={'finite_difference':fd,'P_exact_reference':rp,'max_error':err}
        check(name+'_envelope_gradient',err<5e-9,max_error=err)
    t=d['small_centered_scale'];sw=[[t*v for v in r] for r in cases['centered23']]
    small={'scale':t,'scaled_matrix':sw,'F':reference_2xm(sw,tau)[0],
           'second_order_prediction':t*t*mean([[v*v for v in r] for r in cases['centered23']])/(2*tau)}
    check('nonzero_small_W_positive',small['F']>0,**small)
    finite=[]
    for name in d['finite_iteration_cases']:
        k=cases[name];rf,rp,_=reference_2xm(k,tau)
        for steps in d['finite_iterations']:
            r=sinkhorn(k,tau,steps,None)
            fd=derivative(lambda x:sinkhorn(x,tau,steps,None)['value'],k,eps)
            tr=sinkhorn(transpose(k),tau,steps,None)
            item={'case':name,'iterations':steps,'approximation':r,'reference_F':rf,
                  'value_error':abs(r['value']-rf),'P_error_to_exact':diff(r['P'],rp),
                  'finite_algorithm_FD':fd,'stop_gradient_vs_finite_algorithm_error':diff(r['P'],fd),
                  'finite_transpose_difference':abs(r['value']-tr['value']),
                  'purpose':'Intentional truncation witness, not a project implementation failure'}
            finite.append(item)
            print('TRUNCATION '+json.dumps({k:v for k,v in item.items() if k not in ('approximation','finite_algorithm_FD')},ensure_ascii=False))
    check('finite_iteration_stop_gradient_counterexample',any(x['stop_gradient_vs_finite_algorithm_error']>1e-4 for x in finite))
    check('finite_iteration_transpose_counterexample',any(x['finite_transpose_difference']>1e-4 for x in finite))
    result={'scope':d['purpose'],'tau':tau,'cases':results,'properties':props,'counterexample':counter,
            'gradients':gradients,'small_W':small,'finite_iterations':finite,'checks':checks,
            'summary':{'checks':len(checks),'passed':sum(r['passed'] for r in checks),
                       'max_reference_value_error':max_reference_error,
                       'max_envelope_FD_error':max_gradient_error,
                       'finite_iteration_witnesses':len(finite)}}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('SUMMARY '+json.dumps(result['summary'],ensure_ascii=False))
    if not all(x['passed'] for x in checks):raise AssertionError('At least one reference check failed; retain output and logs.')
if __name__=='__main__':main()
