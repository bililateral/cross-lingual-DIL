from pathlib import Path
import difflib,json,hashlib
r=Path('/mnt/data/bge_continual_external_audit');p=r/'scripts/independent_checks_v1.py';s=p.read_text()
s=s.replace('def scalar_objective(z,y=Y):','def scalar_objective(z,y=Y,hard_threshold=None):')
s=s.replace("scale=1/(28*len(pos)*5);vals['hard']+=sp(z[n]-z[p])*scale\n    d=sig(z[n]-z[p])*scale", "scale=1/(28*len(pos)*5);delta=z[n]-z[p]\n    linear=hard_threshold is not None and delta>hard_threshold\n    vals['hard']+=(delta if linear else sp(delta))*scale\n    d=(1. if linear else sig(delta))*scale")
s=s.replace("vals,grad=scalar_objective(z);t=torch.tensor(z,dtype=torch.float64,requires_grad=True);", "exact_vals,exact_grad=scalar_objective(z)\n   vals,grad=scalar_objective(z,hard_threshold=20.);t=torch.tensor(z,dtype=torch.float64,requires_grad=True);")
s=s.replace("row={'case':name}","row={'case':name,'exact_mathematical_vs_runtime_softplus':{k:{'value_difference':vals[k]-exact_vals[k],'gradient_max_difference':float(np.max(abs(grad[k]-exact_grad[k])))} for k in vals},'runtime_threshold_explicit':20}")
s=s.replace("class SmallScore(torch.nn.Module):",'''def mixed_target_rounding_correction(z):
 """Value only: actual runner makes float32 labels even for this double toy.
 Exact Bernoulli derivative remains double (measured in diagnosis), whereas
 BCE values are per-element float32 before reduction. Do not fake gradients.
 """
 term=torch.stack([torch.logaddexp(z.new_zeros(()),v)-int(y)*v for v,y in zip(z,Y)])
 return float(term.detach().float().mean())-float(term.detach().mean())

class IndependentHandmadeBudget:
 """No formal budget assertion; fake monitoring interface for actual tiny path."""
 def check(self,*args,**kwargs): pass
 def state(self): return {'independent_scope':'handmade CPU; not formal resource measurement'}

class SmallScore(torch.nn.Module):''')
s=s.replace("torch.manual_seed(520);pc=ref(current);loss=torch_reference(pc)","torch.manual_seed(520);pc=ref(current);loss=torch_reference(pc);value_correction=mixed_target_rounding_correction(pc)")
s=s.replace("torch.manual_seed(521);ph=ref(old);loss=loss+torch_reference(ph)","torch.manual_seed(521);ph=ref(old);loss=loss+torch_reference(ph);value_correction+=mixed_target_rounding_correction(ph)")
s=s.replace("self.assertLess(err,1e-10);self.assertAlmostEqual(got['total'],float(loss.detach()),places=10)","expected_logged=float(loss.detach())+value_correction\n   self.assertLess(err,1e-10);self.assertLessEqual(abs(got['total']-expected_logged),1e-12)")
s=s.replace("'total_error':abs(got['total']-float(loss.detach()))", "'total_error':abs(got['total']-expected_logged),'mathematical_double_total':float(loss.detach()),'float32_target_value_correction':value_correction")
s=s.replace("budget=submitted.HandmadeBudget()","budget=IndependentHandmadeBudget()")
s=s.replace("m.calibration.objective(optimum,z,y)","m.calibration.loss_gradient(optimum,z,y)")
# Avoid claiming use of Newton when the independent exact optimum is analytic.
s=s.replace('independent Newton in (a,b), never formal stage mapping.','independent closed-form optimum in (a,b), never formal stage mapping.')
p2=r/'scripts/independent_checks_v2.py';p2.write_text(s)
(r/'scripts/independent_v1_to_v2.diff').write_text(''.join(difflib.unified_diff(p.read_text().splitlines(True),s.splitlines(True),fromfile=p.name,tofile=p2.name)))
record={'revision':'v1 -> v2','project_files_modified':False,'metric_tolerance_unchanged':1e-12,'diagnosis':'outputs/reference_v1_diagnosis.json','changes':['Hard-loss scalar reference uses explicit documented threshold20 for matching runtime; exact mathematical differences separately retained, no tolerance loosening.','Double-valued toy preserves exact independent gradient graph; scalar reference explicitly accounts for runner float32 target BCE output rounding measured in diagnosis.','Own budget fixture now implements runner-required state interface, without claiming formal resource enforcement.','Correct function lookup objective -> loss_gradient, confirmed exact source function.'], 'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p,p2)}}
(r/'outputs/reference_revision_v2.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
