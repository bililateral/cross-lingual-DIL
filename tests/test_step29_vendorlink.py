"""Handwritten-only Linux qualification; real tiny BERT, no downloaded weights."""
import copy
import csv
from dataclasses import asdict
import json
import io
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import tarfile
import unittest
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizer

import step29_vendorlink_data as d
import step29_vendorlink_methods as m
import step29_vendorlink_run as run
import step29_vendorlink_prepare as prepare


def fixture(domain="A", split="fit", component_sizes=(2, 2, 2, 2)):
    views = []
    for identity, count in enumerate(component_sizes):
        for view in range(count):
            records = tuple(d.Ad(f"{domain}-{identity}-{view}-{i}", f"blue item {identity}",
                                f"text {view} sample {i}") for i in range(2+(view % 2)))
            views.append(d.View(f"{domain}:{identity}:{view}", records, f"{domain}:{identity}"))
    return d.Group(domain+":group", domain, split, tuple(views))


def model_for(method, directory):
    vocab = directory/"vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "blue", "item", "text",
                              "sample"]+[str(i) for i in range(20)])+"\n")
    tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=12, max_position_embeddings=32,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=.1)
    torch.manual_seed(41)
    model = m.Model(BertModel(config), tokenizer, m.Settings(method, channel_tokens=16, vendor_tokens=24,
                                                          microbatch=64, vendor_batch=100))
    return model


def independent_pool(table, slots):
    # Direct lists of record pairs, no matrix/bucket implementation shared with production.
    lookup = {(i, j): k for k, (i, j) in enumerate((i, j) for i in range(len(slots)) for j in range(i+1, len(slots)))}
    outputs = []
    for a in range(max(slots)+1):
        for b in range(a+1, max(slots)+1):
            values = torch.stack([table[lookup[tuple(sorted((i, j)))]] for i, slot in enumerate(slots) if slot == a
                                  for j, other in enumerate(slots) if other == b]).double()
            # Modest handwritten inputs permit an independent direct exp reference.
            outputs.append((.5*values.mean()+.5*values.exp().mean().log()).to(table.dtype))
    return torch.stack(outputs)


def independent_loss(scores, group, weight):
    pairs = [(i, j) for i in range(len(group.views)) for j in range(i+1, len(group.views))]
    lookup = {pair: k for k, pair in enumerate(pairs)}
    labels = group.labels()
    bce = sum(F.softplus(-s) if y else F.softplus(s) for s, y in zip(scores, labels))/len(labels)
    ranks, hard = [], []
    for q, view in enumerate(group.views):
        pos = [scores[lookup[tuple(sorted((q, j)))]] for j, other in enumerate(group.views)
               if q != j and view.identity == other.identity]
        neg = [scores[lookup[tuple(sorted((q, j)))]] for j, other in enumerate(group.views)
               if view.identity != other.identity]
        ranks.append(sum(torch.log(1+sum((n-p).exp() for n in neg)) for p in pos)/len(pos))
        top = sorted(range(len(neg)), key=lambda j: (-float(neg[j].detach()), j))[:5]
        hard.append(sum(F.softplus(neg[j]-p) for p in pos for j in top)/(len(pos)*5))
    return bce+weight*sum(ranks)/len(ranks)+.5*sum(hard)/len(hard)


class Qualification(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ["VENDORLINK_TEST_ROOT"])
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def evidence(self, name, value):
        run.write_json(Path(os.environ["VENDORLINK_TEST_ROOT"])/(name+".json"), value)

    def test_public_source_adapter(self):
        headers = ['hash', 'market_name', 'item_link', 'vendor_name', 'name', 'description']
        early = [['1', 'AlphaBay', 'url1', 'alice', 'first title', 'first description'],
                 ['2', 'AlphaBay', 'url2', 'bob', 'ambiguous title', 'ambiguous description'],
                 ['3', 'Dream', 'wrong', 'other', 'wrong market', 'wrong text'],
                 ['4', 'AlphaBay', 'bad', 'bad', 'extra', 'bad', 'extra column']]
        later = [['1', 'AlphaBay', 'url1', 'alice', 'later title', 'later description'],
                 ['2', 'AlphaBay', 'url2', 'charlie', 'changed owner', 'changed owner text']]
        dream = [['5', 'Dream', 'dream1', 'dream_vendor', 'dream title', 'dream description']]
        grams, agora = self.root/'grams.tar.xz', self.root/'agora.zip'
        with tarfile.open(grams, 'w:xz') as archive:
            # Later snapshot first in archive: date, not traversal, must choose the retained text.
            for name, records in [('root/2016-01-02/Alpha.csv', later),
                                  ('root/2016-01-01/Alpha.csv', early),
                                  ('root/2016-01-01/Dream.csv', dream)]:
                text = io.StringIO(newline='')
                writer = csv.writer(text)
                writer.writerows([headers, *records])
                payload = text.getvalue().encode()
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        with zipfile.ZipFile(agora, 'w') as archive:
            text = io.StringIO(newline='')
            csv.writer(text).writerows([['Vendor', ' Item', ' Item Description'],
                                        ['00123', 'agora title', 'agora description'],
                                        ['empty', 'empty title', '']])
            archive.writestr('Agora.csv', text.getvalue().encode('ISO-8859-1'))
        rows, report = prepare.read_sources(grams, agora)
        self.assertEqual(len(rows), 3)
        by_market = {r['market']: r for r in rows}
        self.assertEqual(by_market['AlphaBay']['title'], 'first title')
        self.assertEqual(by_market['Agora']['vendor'], '00123')
        self.assertEqual(report['markets']['AlphaBay']['ambiguous_account_listings_removed'], 1)
        self.assertEqual(report['markets']['AlphaBay']['wrong_market_rows'], 1)
        self.assertEqual(report['markets']['AlphaBay']['malformed_rows'], 1)
        self.assertEqual(report['markets']['Agora']['missing_required_rows'], 1)
        self.evidence('source_adapter', {'canonical_rows': len(rows), 'earliest_not_archive_order': True,
            'ambiguous_listing_excluded': True, 'numeric_handle_string_preserved': True,
            'wrong_market_and_bad_rows_excluded': True, 'handwritten_only': True})

    def test_joint_preview_and_actual_builder_agree(self):
        rows = [{'market': market, 'vendor': f'handle{i}', 'title': f'{market} product {i} {j}',
                 'description': f'{market} text {i} {j}'}
                for market in ('A', 'B', 'C') for i in range(90) for j in range(20)]
        rows[1800] = {**rows[0], 'market': 'B'}
        rows.append(dict(rows[0]))
        options = dict(split_seed=20261029, train_fraction=.7, dev_fraction=.15,
                       records_per_view=2, identities_per_group=8)
        preview = d.preview_counts(rows, **options)
        self.assertEqual([preview['markets'][m]['eligible_accounts'] for m in 'ABC'], [89, 89, 90])
        self.assertFalse(preview['views_or_qrels_created'])
        payload, labels, report = d.build_views(rows, **options)
        for key, values in payload.items():
            part = preview['partitions'][key]
            self.assertEqual(part['groups'], len(values))
            self.assertEqual(part['views'], sum(len(g['views']) for g in values))
            self.assertEqual(part['selected_ads'], sum(len(v['records']) for g in values for v in g['views']))
        self.assertEqual(preview['rejected'], report['rejected'])
        self.evidence('joint_preview', {'expected_eligible_accounts': [89, 89, 90],
            'cross_market_shared_text_removes_both_accounts_at_threshold': True,
            'preview_matches_actual_builder_groups_views_ads': True, 'handwritten_only': True})

    def test_data_views_splits_and_no_label_scoring(self):
        rows = [{"market": market, "vendor": f"vendor{i}", "title": f"product {i} {j} {market}",
                 "description": f"vendor{i} item description {j} {market}"}
                for market in ("A", "B", "C") for i in range(90) for j in range(20)]
        rows += [dict(rows[0]), {**rows[0], "vendor": "copycat"}]
        payload, labels, report = d.build_views(rows, split_seed=9, train_fraction=.5, dev_fraction=.25,
                                                records_per_view=2, identities_per_group=4)
        groups = [d.Group.from_dict(g) for part in payload.values() for g in part]
        uid_split = {}
        for group in groups:
            for view in group.views:
                self.assertTrue(view.identity is not None if group.split == "fit" else view.identity is None)
                for ad in view.records:
                    self.assertNotIn(ad.uid, uid_split)
                    uid_split[ad.uid] = group.split
                    self.assertNotRegex(ad.description, r"vendor\d+")
            self.assertEqual(len(labels[group.uid]), len(group.views)*(len(group.views)-1)//2)
            # Handwritten product number is private fixture truth, independent of output order.
            expected = [int(a.records[0].title.split()[1] == b.records[0].title.split()[1])
                        for i, a in enumerate(group.views) for b in group.views[i+1:]]
            self.assertEqual(labels[group.uid], expected)
        self.assertGreater(report["rejected"]["shared_exact_text"], 0)
        blind = fixture(split="dev").blind()
        with self.assertRaises(ValueError):
            d.assignment(blind, 1)
        model = model_for("H_full", self.root)
        scores = run.evaluate(model, [blind])
        with self.assertRaises(ValueError):
            run.evaluate(model, [fixture()])
        self.assertEqual(len(scores[blind.uid]["scores"]), len(blind.views)*(len(blind.views)-1)//2)
        self.evidence("data", {**report, "blind_regroup_rejected": True, "fit_evaluation_rejected": True,
                               "blind_scoring_uses_no_identity": True})

    def test_blind_alignment_and_ties(self):
        source = fixture(split="dev")
        aligned, blind = source.evaluation_order(), source.blind()
        self.assertEqual(blind, aligned.blind())
        self.assertTrue(all(v.identity is None for v in blind.views))
        self.assertTrue(all(len(v.uid) == 64 and ":" not in v.uid for v in blind.views))
        # Recover labels here only from private fixture membership, not exported IDs.
        private = {r.uid: v.identity for v in source.views for r in v.records}
        expected = [int(private[a.records[0].uid] == private[b.records[0].uid])
                    for i, a in enumerate(blind.views) for b in blind.views[i+1:]]
        self.assertEqual(expected, aligned.labels())
        self.assertNotEqual(expected, [int(i//2 == j//2) for i in range(8) for j in range(i+1, 8)])
        self.assertEqual(source.blind(), d.Group(source.uid, source.domain, source.split, source.views[::-1]).blind())
        # Relabel identities and old view UIDs: neither may affect public IDs/order.
        relabeled = d.Group(source.uid, source.domain, source.split, tuple(
            d.View("changed-"+str(i), v.records, "other-"+v.identity) for i, v in enumerate(source.views)))
        self.assertEqual(blind, relabeled.blind())
        ids = [v.uid for v in blind.views]
        scores = [0.]*len(expected)
        ap = run.query_ap(scores, expected, ids)
        reference = []
        for q, view in enumerate(aligned.views):
            ranked_ids = sorted(v.uid for v in aligned.views if v.uid != view.uid)
            partner = next(v.uid for v in aligned.views if v.uid != view.uid and v.identity == view.identity)
            reference.append(1/(ranked_ids.index(partner)+1))
        self.assertEqual(ap, reference)
        # Permuting storage changes neither score/label pairing nor AP by query ID.
        permutation = [5, 0, 7, 2, 1, 6, 3, 4]
        other = [aligned.views[i] for i in permutation]
        other_labels = [int(a.identity == b.identity) for i, a in enumerate(other) for b in other[i+1:]]
        other_ap = run.query_ap(scores, other_labels, [v.uid for v in other])
        self.assertEqual(dict(zip(ids, ap)), dict(zip([v.uid for v in other], other_ap)))
        self.evidence("blind_ties", {"qrels_aligned": True, "identity_and_old_uid_independent": True,
            "storage_permutation_invariant": True, "all_equal_score_ap": ap,
            "fixed_tie_rule": "opaque_view_id", "uniform_tie_expectation_not_a_per_instance_target": True})

    def test_vendor_representation_and_diagonal_reference(self):
        layers = tuple(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)/(k+2) for k in range(5))
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        embedding = m.vendor_embeddings(layers, mask)
        expected = torch.stack([sum(layer[q, :int(mask[q].sum())].sum(0) for layer in layers[-4:])/mask[q].sum()
                                for q in range(2)])
        torch.testing.assert_close(embedding, expected)
        perturbed = [v.clone() for v in layers]
        for value in perturbed:
            value[0, 2] = 9999
        torch.testing.assert_close(m.vendor_embeddings(perturbed, mask), embedding)
        z = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.], [1., 0., 1.]])
        slots = [0, 0, 1, 1, 1]
        cosine = lambda u, v: torch.dot(u, v)/(u.norm()*v.norm())
        cross = sum(cosine(u, v) for u in z[:2] for v in z[2:])/6
        a = sum(cosine(u, v) for u in z[:2] for v in z[:2])/4
        b = sum(cosine(u, v) for u in z[2:] for v in z[2:])/9
        actual = m.vendor_scores(z, slots)[0]
        torch.testing.assert_close(actual, 2*cross/(a+b))
        wrong = 2*cross/((a*4-2)/2+(b*9-3)/6)
        self.assertGreater(float((actual-wrong).abs()), .1)
        self.evidence("vendor_math", {"normalized_score": float(actual), "omitted_diagonal_difference": float((actual-wrong).abs()),
            "masked_last_four_sum_mean": True, "author_token_reduction_verified": False,
            "token_reduction_user_approved_assumption": True})

    def test_h_math_and_missing_operations(self):
        errors = {}
        for sizes in ((2, 2, 2, 2), (2,)*8+(3,)*4):
            group = fixture(component_sizes=sizes)
            n = sum(len(v.records) for v in group.views)
            table = torch.sin(torch.arange(n*(n-1)//2, dtype=torch.float64)*.7).requires_grad_()
            teacher = (table.detach()+torch.cos(torch.arange(len(table), dtype=torch.float64)*.4)).clone()
            settings = m.Settings("H_full")
            natural, regrouped = d.assignment(group), d.assignment(group, 19)
            self.assertEqual(sorted(natural), sorted(regrouped))
            for position, slot in enumerate(regrouped):
                self.assertEqual(group.views[slot].identity, group.views[natural[position]].identity)
            actual, _ = m.h_objective(table, group, 19, settings, teacher, True)
            values = [independent_pool(table, s) for s in (natural, regrouped)]
            supervised = sum(independent_loss(v, group, settings.lambda_rank) for v in values)/2
            target = independent_pool(teacher, natural)
            expected = .1*supervised+.5*(values[0]-target).square().mean()
            ga, = torch.autograd.grad(actual, table, retain_graph=True)
            ge, = torch.autograd.grad(expected, table, retain_graph=True)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)
            erased, _ = m.h_objective(table, group, 19, m.Settings("H_ER"), teacher, True)
            gz, = torch.autograd.grad(erased, table)
            self.assertGreater(float((ga-gz).abs().max()), 1e-6)
            errors[str(len(group.views))] = {"gradient_max_error": float((ga-ge).abs().max()),
                                           "remove_kd_gradient_change": float((ga-gz).abs().max())}
        self.evidence("h_math", errors)

    def test_actual_vjp_updates_and_full_restore(self):
        # 35 records / 595 pairs / 70 channel texts cross BOTH actual block boundaries.
        group, history = fixture(component_sizes=(2,)*7), fixture("B", component_sizes=(2,)*7)
        model = model_for("H_full", self.root)
        teacher = model.teacher(history)+.6
        initial = copy.deepcopy(model)
        direct = copy.deepcopy(model)
        optimizer, other = m.make_optimizer(model), m.make_optimizer(direct)
        step = 91
        m.update_h(model, optimizer, group, history, teacher, step)
        direct.train()
        other.zero_grad(set_to_none=True)
        for role, sample, reference, is_history in (("current", group, None, False), ("history", history, teacher, True)):
            rng = d.seed_for(step, role)
            torch.manual_seed(rng)
            values = direct.table(direct.records(sample))
            loss, _ = m.h_objective(values, sample, d.seed_for(rng, "regroup"), direct.settings, reference, is_history)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.)
        other.step()
        for p, q in zip(model.parameters(), direct.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
        encoder_change = float((model.encoder.embeddings.word_embeddings.weight-initial.encoder.embeddings.word_embeddings.weight).detach().abs().max())
        head_change = float((model.head[0].weight-initial.head[0].weight).detach().abs().max())
        self.assertGreater(encoder_change, 1e-7)
        self.assertGreater(head_change, 1e-7)
        memory = m.Memory(5, model.settings)
        memory.retain([history], 1, model)
        path = self.root/"full.pt"
        identity = {"handwritten": True}
        run.save_state(path, model, optimizer, memory, {"updates": 1}, identity)
        expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
        m.update_h(model, optimizer, group, history, teacher, 92)
        expected = copy.deepcopy(model.state_dict())
        expected_adam = copy.deepcopy(optimizer.state_dict())
        restored = model_for("H_full", self.root)
        restored_memory = m.Memory(5, restored.settings)
        restored_optimizer, cursor = run.load_state(path, restored, restored_memory, identity)
        self.assertEqual((random.random(), float(np.random.random())), expected_rng[:2])
        torch.testing.assert_close(torch.rand(3), expected_rng[2], atol=0, rtol=0)
        m.update_h(restored, restored_optimizer, group, history, teacher, 92)
        for key, value in expected.items():
            torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
        for key, state in expected_adam["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(restored_optimizer.state_dict()["state"][key][field], value, atol=0, rtol=0)
        self.evidence("actual_update_restore", {"tiny_random_bert": True, "encoder_change": encoder_change,
            "head_change": head_change, "restored_next_update_exact": True, "direct_autograd_agrees": True,
            "records": 35, "record_pairs": 595, "channel_texts": 70, "head_block": 512, "microbatch": 64,
            "pretrained_weights_loaded": False})

    def test_four_actual_trajectories_memory_and_future_access(self):
        evidence = {}
        for method in m.METHODS:
            model = model_for(method, self.root)
            before = model.encoder.embeddings.word_embeddings.weight.detach().clone()
            loaded = []
            def load(domain):
                self.assertEqual(domain, "ABC"[len(loaded)])
                loaded.append(domain)
                return [fixture(domain)]
            job = self.root/method
            result = run.execute(model, load, job, order="ABC", seed=7, epochs=1,
                                 identity={"scope": "handwritten", "method": method})
            self.assertEqual(loaded, ["A", "B", "C"])
            self.assertTrue(result["complete"])
            self.assertEqual(result["updates"], 3)
            self.assertGreater(float((model.encoder.embeddings.word_embeddings.weight-before).abs().max()), 0)
            states = [torch.load(job/f"stage{i}.pt", weights_only=False) for i in (1, 2, 3)]
            expected_steps = 1 if method == "VendorLink_SEQ" else 3
            self.assertTrue(all(int(value["step"]) == expected_steps for value in states[-1]["optimizer"]["state"].values()))
            if method in ("H_ER", "H_full"):
                first = states[0]["memory"]["entries"][0]
                survivors = {v["group"]["uid"]: v for v in states[1]["memory"]["entries"]}
                torch.testing.assert_close(first["teacher"], survivors[first["group"]["uid"]]["teacher"], atol=0, rtol=0)
                self.assertEqual(survivors[first["group"]["uid"]]["origin"], 1)
            else:
                self.assertFalse(states[-1]["memory"]["entries"])
            logs = [json.loads(line) for line in (job/"updates.jsonl").read_text().splitlines()]
            if method == "H_ER":
                self.assertEqual(logs[-1]["history"]["distillation"], 0)
            if method == "H_full":
                self.assertGreater(logs[-1]["history"]["distillation"], 0)
            if method == "VendorLink_SEQ":
                self.assertTrue(all(c.startswith("C:") for c in model.classes))
                self.assertTrue(all("cross_entropy" in log for log in logs))
                probe = model_for(method, self.root)
                probe.start_vendor_stage(sorted({v.identity for v in fixture().views}), 15)
                probe_optimizer = m.make_optimizer(probe)
                before_head = probe.classifier.weight.detach().clone()
                records = [r for v in fixture().views for r in v.records]
                identities = [v.identity for v in fixture().views for r in v.records]
                m.update_vendor(probe, probe_optimizer, records, identities, 17)
                self.assertGreater(float((probe.classifier.weight-before_head).detach().abs().max()), 1e-7)
            resumed = model_for(method, self.root)
            loaded_after_recovery = []
            def resume_load(domain):
                loaded_after_recovery.append(domain)
                return [fixture(domain)]
            run.execute(resumed, resume_load, self.root/(method+"_resumed"), order="ABC", seed=7, epochs=1,
                identity={"scope": "handwritten", "method": method}, resume=job/"stage1.pt")
            self.assertEqual(loaded_after_recovery, ["B", "C"])
            recovered_job = self.root/(method+"_resumed")
            completion = json.loads((recovered_job/"completion.json").read_text())
            self.assertTrue(all((recovered_job/p).is_file() for p in completion["stage_endpoints"]))
            self.assertEqual(run.file_sha256(job/"stage1.pt"), run.file_sha256(recovered_job/"stage1.pt"))
            self.assertEqual(completion["inherited_endpoints"][0]["sha256"], run.file_sha256(job/"stage1.pt"))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[key], value, atol=0, rtol=0)
            blind = fixture("A", "dev").blind()
            scores = run.evaluate(model, [blind])[blind.uid]["scores"]
            ap = run.query_ap(scores, fixture("A", "dev").evaluation_order().labels(), [v.uid for v in blind.views])
            self.assertEqual(len(ap), len(blind.views))
            self.assertTrue(all(0 <= v <= 1 for v in ap))
            evidence[method] = {"loaded_domains": loaded, "updates": result["updates"],
                "history_members": len(states[-1]["memory"]["entries"]), "blind_pairs": len(scores),
                "stage_boundary_resume_exact": True, "recovered_stage_loads": loaded_after_recovery,
                "all_recovered_endpoints_exist": True, "inherited_endpoint_same_bytes": True}
        self.evidence("actual_four_trajectories", evidence)


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("Research qualification must use existing Linux py310")
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
