"""Resolve exact evidence gaps in the two saved audits, without fabricating truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import html
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time

from step33_vendorlink_alias_audit import ROOT, rows_sql, digest, write_json, write_jsonl


def run(prior, followup, out):
    started=time.monotonic(); os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(600,600)); signal.alarm(600)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    out.mkdir(parents=True,exist_ok=False); evidence=out/'evidence'; evidence.mkdir()
    accounts={r['uid']:r for r in map(json.loads,(prior/'evidence/accounts.jsonl').open())}
    registry=list(map(json.loads,(prior/'evidence/registry_matches.jsonl').open()))
    groups=list(map(json.loads,(prior/'evidence/shared_key_candidates.jsonl').open()))
    links=list(map(json.loads,(followup/'evidence/grams_link_candidates.jsonl').open()))
    bodies=list(map(json.loads,(followup/'evidence/original_text_candidates.jsonl').open()))
    lookup={g['link_group']:{v['account_uid'] for v in g['members']} for g in links}
    identical=sum({v['account_uid'] for v in g['members']}==lookup.get(g['key_row_id']) for g in groups)
    peers=defaultdict(set)
    for g in groups:
        members={v['account_uid'] for v in g['members']}
        for a in members: peers[a].update(members-{a})
    marker=Counter(); claim_accounts=defaultdict(set); directed=defaultdict(list)
    signature_fragments=[]; raw_bounded_keys=[]
    for n,r in enumerate(bodies):
        text=html.unescape(r['raw_title']+'\n'+r['raw_description'])
        key_begins=list(re.finditer(r'BEGIN\s+PGP\s+PUBLIC\s+KEY\s+BLOCK',text,re.I))
        key_ends=list(re.finditer(r'END\s+PGP\s+PUBLIC\s+KEY\s+BLOCK',text,re.I))
        if key_begins: marker[r['market']+':key_begin']+=1
        if key_begins and any(e.start()>key_begins[0].end() for e in key_ends):
            marker[r['market']+':bounded_key_words']+=1
            raw_bounded_keys.append({'body_index':n,'account_uid':r['account_uid'],'source':r['source'],
                'has_standard_armor':bool(r['public_keys']), 'source_lost_armor_dashes':not bool(re.search(r'-{5}BEGIN PGP PUBLIC KEY BLOCK-{5}',text)),
                'raw_line_breaks':r['raw_description'].count('\n'), 'label':'BOUNDED_WORDS_NOT_DECODED_KEY'})
        if re.search(r'BEGIN\s+PGP\s+SIGNED\s+MESSAGE',text,re.I):
            ending=bool(re.search(r'END\s+PGP\s+SIGNATURE',text,re.I))
            marker[r['market']+':signed_message_begin']+=1
            marker[r['market']+':signed_message_with_end_words']+=ending
            signature_fragments.append({'body_index':n,'account_uid':r['account_uid'],'source':r['source'],
                'length':len(text),'has_end_signature_words':ending,'raw_description':r['raw_description'],
                'label':'INCOMPLETE_SIGNATURE_NO_CONTROL_PROOF' if not ending else 'REQUIRES_CRYPTOGRAPHIC_VERIFICATION'})
        if r['claim_contexts']:
            claim_accounts[r['market']].add(r['account_uid'])
            for peer in peers.get(r['account_uid'],[]):
                peer_name=accounts[peer]['vendor']
                if peer_name.casefold()==r['vendor'].casefold(): continue
                for context in r['claim_contexts']:
                    if re.search(r'(?<!\w)'+re.escape(peer_name)+r'(?!\w)',context,re.I):
                        directed[(r['account_uid'],peer)].append({'body_index':n,'source':r['source'],'context':context})
    mutual={tuple(sorted((a,b))) for a,b in directed if (b,a) in directed}
    write_jsonl(evidence/'bounded_key_formats.jsonl',raw_bounded_keys)
    write_jsonl(evidence/'signature_fragments.jsonl',signature_fragments)
    write_jsonl(evidence/'different_name_claim_candidates.jsonl',({'from_account':a,'to_account':b,'evidence':v,'reciprocal':(b,a) in directed,'label':'UNSIGNED_CLAIM_NOT_VERIFIED_CONTROL'} for (a,b),v in directed.items()))

    # Diagnose five nonzero gpg exits instead of treating partial fingerprints as validity.
    failures={g['key_row_id']:g for g in groups if g.get('key_parse',{}).get('returncode') != 0}
    failure_details=[]; gpg_home=evidence/'gpg'; gpg_home.mkdir(mode=0o700)
    cols=['id','alias','fingerprint_f','fingerprint','public_key','vendor_ids','user_hash','review_count','star_rate']
    for number,r in rows_sql(ROOT/'3z669jwe.sql',cols,Counter()):
        if r['id'] not in failures: continue
        result=subprocess.run(['gpg','--batch','--no-options','--homedir',str(gpg_home),'--with-colons','--fingerprint','--show-keys'],
            input=bytes.fromhex(r['public_key'][2:]),capture_output=True,timeout=10)
        failure_details.append({'key_row_id':r['id'],'source_line':number,'returncode':result.returncode,
            'stderr':result.stderr.decode(errors='replace'),'stdout':result.stdout.decode(errors='replace'),
            'label':'EXCLUDE_FROM_CRYPTO_VALIDATED_KEYS'})
    write_jsonl(evidence/'gpg_failure_details.jsonl',failure_details)

    # Capacity is conditional on accepting source-defined weak relations, not a gold split.
    ambiguous={a for a,n in Counter(v['account_uid'] for v in registry if v['account_uid']).items() if n>1}
    clean_groups=[]
    for g in groups:
        if g['imposter_marked_members'] or g['key_row_id'] in failures: continue
        if not g.get('key_parse',{}).get('declared_matches_primary'): continue
        members=[v['account_uid'] for v in g['members'] if v['account_uid'] not in ambiguous and v['match_kind']=='exact_market_name' and accounts[v['account_uid']]['distinct_texts']>=2]
        if len(members)>=2:
            clean_groups.append({'group_id':g['key_row_id'],'accounts':members,
                'different_names':len({accounts[a]['vendor'].casefold() for a in members})>1,
                'markets':sorted({accounts[a]['market'] for a in members}),
                'label':'FILTERED_WEAK_RELATION_CANDIDATE_NOT_GOLD'})
    write_jsonl(evidence/'filtered_weak_relation_candidates.jsonl',clean_groups)
    summary={'status':'EVIDENCE_GAPS_MEASURED_CONTEXTUAL_ADJUDICATION_REQUIRED','observed_server':datetime.now().astimezone().isoformat(),
        'script_sha256':digest(Path(__file__)),'prior_summary_sha256':digest(prior/'summary.json'),'followup_summary_sha256':digest(followup/'summary.json'),
        'same_group_source_dependency':{'pgp_groups':len(groups),'same_id_and_membership_in_grams_link':identical,'independent_corroborations_claimed':0},
        'format_followup':dict(marker),'bounded_word_key_records':len(raw_bounded_keys),
        'bounded_word_key_accounts':len({r['account_uid'] for r in raw_bounded_keys}),
        'signed_message_fragment_records':len(signature_fragments),'signed_message_fragment_accounts':len({r['account_uid'] for r in signature_fragments}),
        'complete_signed_messages_with_loose_end_marker':sum(r['has_end_signature_words'] for r in signature_fragments),
        'claim_accounts_by_market':{m:len(v) for m,v in claim_accounts.items()},
        'different_name_directed_claim_candidate_pairs':len(directed),'mutual_different_name_claim_candidate_pairs':len(mutual),
        'gpg_nonzero_exit_keys':len(failure_details),'gpg_nonzero_exit_keys_excluded':True,
        'conditional_filtered_weak_relation_capacity':{'groups':len(clean_groups),'accounts':len({a for g in clean_groups for a in g['accounts']}),
            'different_name_groups':sum(g['different_names'] for g in clean_groups),
            'groups_by_markets':dict(Counter('+'.join(g['markets']) for g in clean_groups)),
            'accounts_by_market':dict(Counter(accounts[a]['market'] for g in clean_groups for a in g['accounts']))},
        'verified_control_pairs':None,'verified_negative_pairs':None,'manual_adjudication_required':True,
        'dataset_ready_for_verified_alias_training_evaluation':False,'formal_partitions_created':False,'training_started':False,
        'interpretation':['Missing adjudication is not a count of zero real aliases.',
            'Bounded key markers must not be reported as zero PGP presence; source-normalized keys are not necessarily cryptographically recoverable.',
            'Grams link, SQL key grouping and HTML profile links share provenance and are not three independent truths.',
            'Unverified groups or distinct keys do not establish reliable negatives required by the unchanged H loss.',
            'Weak-label development would require an explicit scientific-scope decision, not a silent substitute for verified alias evaluation.',
            'Keyword claim recall is limited; saved candidates require contextual interpretation, not automatic positives.'],
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prior',type=Path,required=True); p.add_argument('--followup',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run(a.prior,a.followup,a.output)
