"""Handwritten independent checks. Execute only on the project Linux py310."""
from __future__ import annotations

import copy
import hashlib
import itertools
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_function_memory as method


def handmade_group(uid: str, records: int = 2, text_length: int = 0) -> method.data.Group:
    controls = [i for i, n in enumerate([3]*4 + [2]*8) for _ in range(n)]
    labels = tuple(int(controls[i] == controls[j]) for i, j in itertools.combinations(range(28), 2))
    def text(i: int, j: int, channel: int) -> str:
        if text_length:
            # CJK characters are individually tokenized by the pinned BGE vocab.
            prefix = f"{'旧' if 'history' in uid else '新'}款{chr(0x4e00+i)}{chr(0x4e40+j)}{chr(0x4e60+channel)}"
            return prefix + "物" * (text_length-len(prefix))
        return f"手写{uid}商品{i}记录{j}通道{channel}。"
    group = method.data.Group(uid, tuple(f"{uid}_s{i:02}" for i in range(28)),
        tuple(tuple((f"{uid}_i{i:02}_{j}", text(i,j,0), text(i,j,1))
                    for j in range(records)) for i in range(28)), labels)
    group.validate()
    return group


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 4)
        self.dropout = torch.nn.Dropout(.3)

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": torch.tensor([[v % 64 for v in hashlib.sha256(t.encode()).digest()[:5]]
                                           for t in texts])}

    def forward(self, batch):
        return {"sentence_embedding": self.dropout(self.embedding(batch["input_ids"])).mean(1)}


def tiny_model():
    torch.manual_seed(41)
    return method.build_model(TinyEncoder(), 16)


def geometry(labels):
    edges = list(itertools.combinations(range(28), 2))
    g = np.eye(378)/378
    for q in range(28):
        pos = [i for i,e in enumerate(edges) if q in e and labels[i]]
        neg = [i for i,e in enumerate(edges) if q in e and not labels[i]]
        for i in pos:
            for j in neg:
                scale = 1/(28*len(pos)*len(neg))
                g[i,i] += scale; g[j,j] += scale
                g[i,j] -= scale; g[j,i] -= scale
    return torch.from_numpy(g)


def random_z(seed=1):
    z = np.random.default_rng(seed).uniform(.05,.8,(129,378)).astype(np.float16)
    z[-1] = 1
    return z


def fixture_memory(model, old, c):
    x = method.reference(model, old, c)
    h,b,k = method.statistics(torch.from_numpy(x),old.labels,method.weight(model).detach())
    memory = method.Memory("ABC")
    memory.h,memory.b,memory.constant,memory.count=h.numpy(),b.numpy(),k,1
    return x,memory


class FunctionMemoryTests(unittest.TestCase):
    def test_explicit_two_teacher_quadratic_transport_and_gradient(self):
        labels=handmade_group("math").labels
        g=geometry(labels)
        self.assertAlmostEqual(float(g.trace()),3,places=12)
        x,y=(torch.tensor(random_z(i),dtype=torch.float64) for i in (1,2))
        w1=torch.linspace(-.07,.09,129,dtype=torch.float64)
        w2=torch.flip(w1,[0])+.03
        h1,b1,c1=method.statistics(x,labels,w1)
        h2,b2,c2=method.statistics(y,labels,w2)
        torch.testing.assert_close(h1,x@g@x.T,rtol=1e-12,atol=1e-12)
        zero=method.historical_loss(x,w1,x,h1,b1,c1,1)
        self.assertLess(abs(float(zero)),1e-12)
        t=method.transport(x,y).detach()
        ht=t@h1@t.T+h2; bt=t@b1+b2; ct=c1+c2
        v=w2+.02
        explicit=((x.T@t.T@v-x.T@w1)@g@(x.T@t.T@v-x.T@w1)
                  +(y.T@v-y.T@w2)@g@(y.T@v-y.T@w2))/6
        actual=(v@ht@v-2*bt@v+ct)/6
        torch.testing.assert_close(actual,explicit,rtol=1e-10,atol=1e-12)
        self.assertGreater(float((bt-ht@w2).norm()),1e-4)
        yy=y.clone().requires_grad_(); ww=v.clone().requires_grad_()
        loss=method.historical_loss(yy,ww,x,h1,b1,c1,1)
        dy,dw=torch.autograd.grad(loss,(yy,ww))
        k=x@x.T/378+.001*torch.eye(129,dtype=torch.float64)
        vv=torch.linalg.solve(k,x@(y.T@v)/378+.001*v)
        tt=torch.linalg.solve(k.T,2*(h1@vv-b1)/3)
        gg=x.T@tt/378
        torch.testing.assert_close(dy,v[:,None]*gg[None,:],rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(dw,y@gg+.001*tt,rtol=1e-10,atol=1e-12)
        self.assertGreater(float((.001*tt).norm()),1e-6)

    def test_quantization_straight_through_overflow_and_checkpoint_mode(self):
        from torch.utils.checkpoint import checkpoint
        h=torch.linspace(.001,1.1,378*128).reshape(378,128).requires_grad_()
        z=method.canonical_features(h)
        self.assertTrue(torch.equal(z[:-1].T,h.detach().half().float()))
        incoming=torch.full_like(h,1.000123)
        (z[:-1].T*incoming).sum().backward()
        self.assertTrue(torch.equal(h.grad,incoming))
        with self.assertRaisesRegex(ValueError,"overflow"):
            method.canonical_features(torch.full((378,128),1e8))
        module=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(4,4),torch.nn.ReLU())
        module[0].dropout_prob=.2
        observed=[]
        def forward(a):
            observed.append(module[0].p)
            return module(a)
        a=torch.ones(8,4,requires_grad=True)
        with method.deterministic_history(module):
            checkpoint(forward,a,use_reentrant=False).sum().backward()
        self.assertGreaterEqual(len(observed),2)
        self.assertEqual(set(observed),{0.})
        self.assertEqual(module[0].p,.3); self.assertEqual(module[0].dropout_prob,.2)
        observed.clear()
        with method.deterministic_history(module):
            value=checkpoint(forward,a,use_reentrant=False)
        try: value.sum().backward()
        except torch.utils.checkpoint.CheckpointError: pass
        self.assertIn(.3,observed)  # This fault would escape a forward-only mode check.

    def test_actual_joint_update_matches_independent_full_gradient_and_adam(self):
        c=method.config(); model=tiny_model()
        current=handmade_group("current"); old=handmade_group("history")
        controls=[i for i,n in enumerate([3]*4+[2]*8) for _ in range(n)]
        controls=controls[5:]+controls[:5]
        labels=tuple(int(controls[i]==controls[j]) for i,j in itertools.combinations(range(28),2))
        old=method.data.Group(old.uid,old.sellers,old.items,labels)
        self.assertNotEqual(current.labels,old.labels)
        x,memory=fixture_memory(model,old,c)
        with torch.no_grad(): model.head[2].weight.add_(.01)
        reference=copy.deepcopy(model)
        opt=method.make_optimizer(model,c); opt_ref=method.make_optimizer(reference,c)
        for group in opt_ref.param_groups: group["lr"]=.001
        opt_ref.param_groups[0]["lr"]=1e-5
        reference.train(); torch.manual_seed(7)
        pred=method.base.logits(reference,current,c,"split_rank")
        current_loss=method.parent.ranking.objectives(pred,torch.tensor(current.labels).float(),.5)["total"]
        with method.deterministic_history(reference):
            torch.manual_seed(9)
            scores,z=method.forward_group(reference,old,c)
            w=method.weight(reference).double(); xx=torch.from_numpy(x).double()
            k=xx@xx.T/378+.001*torch.eye(129,dtype=torch.float64)
            v=torch.linalg.solve(k,xx@(z.double().T@w)/378+.001*w)
            # Independent explicit original teacher residual, no production H/b/c quadratic.
            teacher_model=tiny_model()
            target=xx.T@method.weight(teacher_model).detach().double()
            residual=xx.T@v-target
            penalty=residual@geometry(old.labels)@residual/3
            history_loss=method.parent.ranking.objectives(scores,torch.tensor(old.labels).float(),.5)["total"]
            objective=current_loss+.1*history_loss+.5*penalty
            params=list(reference.parameters())
            grad=torch.autograd.grad(objective,params,retain_graph=True)
            variants={"missing_D":current_loss+.1*history_loss,
                      "wrong_coefficient":current_loss+.1*history_loss+.1*penalty,
                      "wrong_labels":current_loss+.1*method.parent.ranking.objectives(scores,torch.tensor(current.labels).float(),.5)["total"]+.5*penalty}
            # Recompute two distinct broken function paths on the same live graph.
            for name,rhs in (("detach_Y",xx@(z.detach().double().T@w)/378+.001*w),
                             ("missing_epsilon_w",xx@(z.double().T@w)/378)):
                bad_v=torch.linalg.solve(k,rhs); error=xx.T@bad_v-target
                variants[name]=current_loss+.1*history_loss+.5*(error@geometry(old.labels)@error/3)
            for name,bad in variants.items():
                badgrad=torch.autograd.grad(bad,params,retain_graph=True)
                distance=max(float((a-b).abs().max()) for a,b in zip(grad,badgrad))
                self.assertGreater(distance,1e-7,name)
            objective.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True); opt_ref.step()
        captured=[]
        def observe(stage):
            if stage=="clip_and_adam": captured.extend(p.grad.clone() for p in model.parameters())
        row=method.optimization_step(model,opt,current,old,x,memory,c,1e-5,
            current_seed=7,history_seed=9,observe_stage=observe)
        for got,wanted in zip(captured,grad): torch.testing.assert_close(got,wanted,rtol=2e-5,atol=2e-7)
        for got,wanted in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(got,wanted,rtol=2e-5,atol=2e-7)
        self.assertEqual(row["adam_step"],1); self.assertGreater(row["function"],0)
        self.assertEqual(model.encoder.dropout.p,.3)

    def test_memory_two_stages_exact_container_and_byte_gate_rollback(self):
        groups=[handmade_group(f"old{i:02}") for i in range(48)]
        new=[handmade_group(f"new{i:02}") for i in range(48)]
        z=random_z(); w=torch.linspace(-.02,.03,129)
        memory=method.Memory("ABC")
        first=memory.consolidate(groups,1,lambda g:z,w)
        self.assertLess(first["serialized_bytes"],1048576)
        before=memory.to_bytes(); restored=method.Memory.from_bytes(before)
        self.assertEqual(restored.to_bytes(),before)
        with mock.patch.object(method,"MAXIMUM_BYTES",len(before)):
            with self.assertRaises(ValueError): memory.begin_stage(2)
        self.assertEqual(memory.to_bytes(),before)
        memory.begin_stage(2)
        for _ in range(288): memory.draw()
        b_old=memory.b.copy(); c_old=memory.constant
        evidence=memory.consolidate(new,2,lambda g:z,w+.07)
        h,b,c=method.statistics(torch.from_numpy(z),new[0].labels,w+.07)
        np.testing.assert_allclose(memory.b,b_old+48*b.numpy(),rtol=1e-10,atol=1e-10)
        self.assertAlmostEqual(memory.constant,c_old+48*c,places=9)
        self.assertGreater(np.linalg.norm(memory.b-memory.h@(w+.07).double().numpy()),1e-3)
        self.assertEqual(memory.count,96); self.assertEqual(memory.reservoir.seen,96)
        self.assertLess(evidence["serialized_bytes"],1048576)
        self.assertEqual(method.Memory.from_bytes(memory.to_bytes()).to_bytes(),memory.to_bytes())


