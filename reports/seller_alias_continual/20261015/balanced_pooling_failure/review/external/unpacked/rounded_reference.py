"""Analytic reference for the SAME two handwritten symmetric 3x3 inputs.
No Sinkhorn, production solver, model, training, or formal data.
Usage: python rounded_reference.py OUTPUT_JSON
For K=[[a,b,c],[b,a,c],[c,c,d]], let s=1/3 and
 beta=exp(2c-d)/(exp(a)+exp(b)).
The optimum has cross entries x=2*beta*s/(3*beta+sqrt(beta*beta+4*beta)),
 P11=P22=(s-x)/(1+exp(b-a)), P12=P21=s-x-P11, P33=s-2*x.
All reference computations use 80 decimal digits; this is not interval arithmetic.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import mpmath as mp
import torch

def main() -> None:
    if len(sys.argv)!=2:raise SystemExit('Usage: python rounded_reference.py OUTPUT_JSON')
    mp.mp.dps=80
    out=[]
    for e in [1e-8,1e-4]:
        p=torch.tensor([[1/3-.05-e,.05,e],[.05,1/3-.05-e,e],[e,e,1/3-2*e]],dtype=torch.float64)
        k64=torch.log(9*p)
        for kind,k in [('float64',k64),('float32',k64.float())]:
            vals=[[mp.mpf(float(k[i,j])) for j in range(3)] for i in range(3)]
            a,b,c,d=vals[0][0],vals[0][1],vals[0][2],vals[2][2]
            s=mp.mpf(1)/3
            beta=mp.exp(2*c-d)/(mp.exp(a)+mp.exp(b))
            x=2*beta*s/(3*beta+mp.sqrt(beta*beta+4*beta))
            u=(s-x)/(1+mp.exp(b-a));v=s-x-u
            plan=[[u,v,x],[v,u,x],[x,x,s-2*x]]
            value=sum(plan[i][j]*(vals[i][j]-mp.log(9*plan[i][j])) for i in range(3) for j in range(3))
            row_error=max(abs(sum(row)-s) for row in plan)
            a0=mp.log(9*u)-a;a2=mp.log(9*(s-2*x))-d
            # Potentials a_i=b_i=half of these diagonal residuals.
            potentials=[a0/2,a0/2,a2/2]
            stationarity=max(abs(mp.log(9*plan[i][j])-vals[i][j]-potentials[i]-potentials[j]) for i in range(3) for j in range(3))
            out.append({'epsilon':e,'input_dtype':kind,'actual_input_K':k.tolist(),'reference_optimal_value':str(value),
                'reference_plan':[[str(v) for v in row] for row in plan],
                'marginal_error_at_80_digits':str(row_error),'stationarity_error_at_80_digits':str(stationarity)})
    Path(sys.argv[1]).write_text(json.dumps({'scope':'SAME_TWO_HANDWRITTEN_EXAMPLES_CLOSED_FORM_REFERENCE_ONLY',
        'production_solver_called':False,'optimizer_calls':0,'repair_implemented':False,
        'precision_decimal_digits':80,'strict_interval_certificate':False,'cases':out},indent=2)+'\n',encoding='utf-8')
    for r in out:print(r['epsilon'],r['input_dtype'],r['reference_optimal_value'])
if __name__=='__main__':main()
