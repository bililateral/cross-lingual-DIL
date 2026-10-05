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
import step28_relation_memory as method


def handmade_group(uid: str, records: int = 2, text_length: int = 0) -> method.data.Group:
    controls = [i for i, n in enumerate([3]*4 + [2]*8) for _ in range(n)]
    labels = tuple(int(controls[i] == controls[j]) for i, j in itertools.combinations(range(28), 2))
    def text(i: int, j: int, channel: int) -> str:
        if text_length:
            # CJK characters are individually tokenized by the pinned BGE vocab.
            prefix = f"商款{chr(0x4e00+i)}{chr(0x4e40+j)}{chr(0x4e60+channel)}"
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


def explicit(scores, labels):
    edges = list(itertools.combinations(range(28), 2))
    identification = sum((s-(2*y-1))**2 for s,y in zip(scores,labels)) / 378
    losses = []
    for q in range(28):
        pos = [scores[i] for i,e in enumerate(edges) if q in e and labels[i]]
        neg = [scores[i] for i,e in enumerate(edges) if q in e and not labels[i]]
        losses.append(sum((p-n-2)**2 for p in pos for n in neg)/(len(pos)*len(neg)))
    return identification + sum(losses)/28


def random_z(seed=1):
    z = np.random.default_rng(seed).uniform(-.8,.8,(32,378)).astype(np.float32)
    z[-1] = 1
    return z


class RelationMemoryTests(unittest.TestCase):
    def test_explicit_query_loss_statistics_and_gradient(self):
        g = handmade_group("formula")
        z = torch.from_numpy(random_z()).double()
        w = torch.linspace(-.3,.4,32,dtype=torch.float64,requires_grad=True)
        h,b,c = method.statistics(z,g.labels)
        loss = method.group_loss(z.T@w,g.labels)
        quadratic = w@h@w-2*b@w+c
        self.assertAlmostEqual(float(loss),float(explicit(z.T@w,g.labels)),places=11)
        self.assertAlmostEqual(float(loss),float(quadratic),places=11)
        grad = torch.autograd.grad(loss,w)[0]
        torch.testing.assert_close(grad,2*(h@w-b),atol=1e-10,rtol=1e-9)
        self.assertAlmostEqual(float(h[-1,-1]),1.)
        self.assertAlmostEqual(float(b[-1]),-169/189)
        self.assertEqual(c,5.)
        # A wrong global 1016-comparison average must disagree with query averaging.
        pairs = method.query_indices(g.labels,"cpu")
        scores = z.T@w
        wrong_rank = torch.cat([(scores[p,None]-scores[n][None,:]-2).square().flatten()
                                 for p,n in pairs]).mean()
        proper_rank = loss-(scores-(2*torch.tensor(g.labels)-1)).square().mean()
        self.assertGreater(abs(float(wrong_rank-proper_rank)),1e-5)

    def test_transport_full_gradients_bias_and_direct_weight_path(self):
        g = handmade_group("grad")
        x = torch.from_numpy(random_z(2)).double()
        y = torch.from_numpy(random_z(3)).double().requires_grad_()
        w = torch.linspace(-.3,.2,32,dtype=torch.float64,requires_grad=True)
        h,b,c = method.statistics(x,g.labels)
        loss = method.historical_loss(y,w,x,h,b,c,1)
        t = method.transport(x,y)
        v = t.T@w
        direct = v@h@v-2*b@v+c
        torch.testing.assert_close(loss,direct,atol=1e-10,rtol=1e-9)
        gy,gw = torch.autograd.grad(loss,(y,w))
        k = x@x.T/378+.001*torch.eye(32,dtype=x.dtype)
        a = 2*(h@v-b)
        torch.testing.assert_close(gw,t@a,atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(gy,torch.outer(w,x.T@torch.linalg.solve(k,a))/378,
                                   atol=1e-10,rtol=1e-9)
        self.assertGreater(float((.001*torch.linalg.solve(k,a)).norm()),1e-5)
        for row,col in [(0,0),(14,129),(31,377)]:
            yp,ym = y.detach().clone(),y.detach().clone()
            yp[row,col] += 1e-5; ym[row,col] -= 1e-5
            fd=(method.historical_loss(yp,w,x,h,b,c,1)-method.historical_loss(ym,w,x,h,b,c,1))/2e-5
            torch.testing.assert_close(fd,gy[row,col],atol=2e-7,rtol=2e-5)
        torch.testing.assert_close(method.transport(x,x),torch.eye(32,dtype=x.dtype),atol=1e-10,rtol=1e-9)
        transform=torch.eye(32,dtype=x.dtype); transform[0,0]=.7
        actual=method.transport(x,transform@x)
        predicted=transform+(torch.eye(32,dtype=x.dtype)-transform)@(.001*torch.linalg.inv(k))
        torch.testing.assert_close(actual,predicted,atol=1e-10,rtol=1e-9)
        # Column permutation is not harmless for the transfer correspondence.
        wrong=method.historical_loss(y,w,x.flip(1),h,b,c,1)
        self.assertGreater(abs(float(wrong-loss)),1e-5)

    def test_proven_blind_direction_and_stage_pushforward(self):
        g=handmade_group("blind")
        edges=list(itertools.combinations(range(28),2))
        x=torch.stack((.5*(2*torch.tensor(g.labels,dtype=torch.float64)-1),torch.ones(378,dtype=torch.float64)))
        indices=[edges.index(e) for e in [(0,3),(0,1),(4,5)]]
        x[0,indices[1]]=0
        y=x.clone(); y[0,indices]+=torch.tensor([.2,-.4,.2])
        w=torch.tensor([1.,0.],dtype=torch.float64)
        h,b,c=method.statistics(x,g.labels)
        torch.testing.assert_close(method.transport(x,y),torch.eye(2,dtype=x.dtype),atol=1e-8,rtol=1e-8)
        self.assertGreater(float(method.group_loss(y.T@w,g.labels)),float(method.group_loss(x.T@w,g.labels)))
        self.assertLess(float(y[0,indices[1]]-y[0,indices[0]]),0)
        t=method.transport(x,y)
        migrated=w@(t@h@t.T)@w-2*(t@b)@w+c
        torch.testing.assert_close(migrated,method.historical_loss(y,w,x,h,b,c,1))

    def test_serial_kernel_matches_joint_and_history_reaches_encoder(self):
        c=method.config(); model=tiny_model(); reference_model=copy.deepcopy(model)
        old,new=handmade_group("old"),handmade_group("new")
        x=method.reference(model,old,c)
        memory=method.Memory("ABC")
        h,b,constant=method.statistics(torch.from_numpy(x),old.labels)
        memory.h,memory.b,memory.constant,memory.count=h.numpy(),b.numpy(),constant,1
        optimizer=method.make_optimizer(model,c)
        joint_optimizer=method.make_optimizer(reference_model,c)
        joint_optimizer.param_groups[0]["lr"]=1e-5
        zc=method.relation_features(reference_model,new,c)
        yh=method.relation_features(reference_model,old,c)
        historical=method.historical_loss(yh,reference_model.weight,x,h,b,constant,1)
        probes=(reference_model.encoder.embedding.weight,reference_model.head[0].weight,reference_model.weight)
        gradients=torch.autograd.grad(historical,probes,retain_graph=True)
        self.assertTrue(all(float(v.norm())>0 for v in gradients))
        (method.group_loss(zc.T@reference_model.weight,new.labels)+historical).backward()
        expected=[p.grad.clone() for p in reference_model.parameters()]
        torch.nn.utils.clip_grad_norm_(reference_model.parameters(),1.)
        joint_optimizer.step()
        original=torch.nn.utils.clip_grad_norm_
        captured=[]
        def capture(params,*args,**kwargs):
            params=list(params); captured.extend(p.grad.clone() for p in params)
            return original(params,*args,**kwargs)
        with mock.patch("torch.nn.utils.clip_grad_norm_",side_effect=capture):
            out=method.optimization_step(model,optimizer,new,old,x,memory,c,1e-5)
        self.assertEqual(out["adam_step"],1)
        for a,b in zip(expected,captured):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)
        for a,b in zip(model.parameters(),reference_model.parameters()):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)
        self.assertEqual(model.encoder.dropout.p,0.)

    def test_lifecycle_byte_budget_restore_and_statistic_migration(self):
        first=[handmade_group(f"first{i:02}") for i in range(48)]
        second=[handmade_group(f"second{i:02}") for i in range(48)]
        z=random_z(7)
        memory=method.Memory("ABC")
        memory.consolidate(first,1,lambda g:z)
        h,b,c=method.statistics(torch.from_numpy(z),first[0].labels)
        np.testing.assert_allclose(memory.h,48*h.numpy(),atol=1e-10)
        payload=memory.to_bytes()
        self.assertLessEqual(len(payload),1048576)
        restored=method.Memory.from_bytes(payload)
        self.assertEqual(restored.to_bytes(),payload)
        memory.begin_stage(2); restored.begin_stage(2)
        for _ in range(288):
            self.assertEqual(memory.draw()[0].uid,restored.draw()[0].uid)
        old_h,old_b=memory.h.copy(),memory.b.copy()
        changed=z.copy(); changed[0]*=.7
        t=method.transport(torch.from_numpy(z),torch.from_numpy(changed)).numpy()
        hc,bc,cc=method.statistics(torch.from_numpy(changed),second[0].labels)
        report=memory.consolidate(second,2,lambda g:changed)
        self.assertEqual(report["reservoir_seen"],96)
        np.testing.assert_allclose(memory.h,t@old_h@t.T+48*hc.numpy(),atol=1e-9)
        np.testing.assert_allclose(memory.b,t@old_b+48*bc.numpy(),atol=1e-9)
        self.assertEqual(memory.constant,480.)
        with self.assertRaises(ValueError): memory.consolidate(second,2,lambda g:changed)
        memory.begin_stage(3)
        memory.reference_keys={k:"wrong" for k in memory.reference_keys}
        with self.assertRaises(ValueError): memory.draw()
        restored.maps={"unbudgeted":"x"*1048576}
        with self.assertRaises(ValueError): restored.to_bytes()

    def test_actual_stage_entry_288_updates_and_continuation(self):
        c=method.config(); model=tiny_model(); opt=method.make_optimizer(model,c)
        memory=method.Memory("ABC")
        first=[handmade_group(f"stage1_{i:02}") for i in range(48)]
        records=method.train_stage(model,opt,first,memory,c,1)
        self.assertEqual(len(records),288)
        self.assertEqual(method.parent.adam_step(opt),288)
        self.assertEqual(memory.count,48)
        self.assertEqual(records[-1]["encoder_lr"],0.)
        saved=copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),memory.to_bytes()
        other=tiny_model(); other.load_state_dict(saved[0])
        other_opt=method.make_optimizer(other,c); other_opt.load_state_dict(saved[1])
        other_memory=method.Memory.from_bytes(saved[2])
        memory.begin_stage(2); other_memory.begin_stage(2)
        new=handmade_group("stage2_first")
        a=method.update(model,opt,new,memory,c,2,1)
        b=method.update(other,other_opt,new,other_memory,c,2,1)
        self.assertEqual(a,b)
        for pa,pb in zip(model.parameters(),other.parameters()):
            torch.testing.assert_close(pa,pb,atol=0,rtol=0)
        self.assertEqual(memory.to_bytes(),other_memory.to_bytes())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
