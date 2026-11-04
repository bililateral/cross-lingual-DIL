"""Approved handwritten GPU probe worker; never opens fit/dev/heldout data.

The parent must first enforce available resources, the pinned deployment and
the shared 1800-worker-second allowance. This file is not a formal launcher.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import BertModel,BertTokenizer

import step28_formal_queue as queue
import step29_vendorlink_data as data
import step29_vendorlink_methods as method
import step29_vendorlink_run as run
import step29_english_abc as policy


def handwritten(domain):
    views=[]
    for identity in range(8):
        for side in range(2):
            # Every channel reaches the existing 256-token bound; 32 ads/group.
            records=tuple(data.Ad(f'{domain}-{identity}-{side}-{i}',
                ('sample blue item '+str(identity)+' ')*90, ('text example '+str(side)+' ')*100) for i in range(2))
            views.append(data.View(f'{domain}:{identity}:{side}',records,f'{domain}:{identity}'))
    return data.Group(domain+':handwritten',domain,'fit',tuple(views))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--permit',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--pooling',choices=('cls','mean_last','mean_last4_sum'),required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    permit=json.loads(args.permit.read_text())
    if (sys.platform!='linux' or permit.get('operation')!='english_abc_handwritten_gpu'
            or permit.get('approved') is not True or permit.get('formal_experiment') is not False
            or permit.get('total_worker_seconds')!=1800 or args.job.exists()):
        raise ValueError('Specific handwritten GPU permit required')
    workspace=Path(permit['workspace']).resolve()
    if not args.job.resolve().is_relative_to(Path(permit['root']).resolve()):
        raise ValueError('Fresh job must belong to this qualification')
    for row in permit['sources']:
        queue.verify(row)
    if str(Path(__file__).resolve()) not in {r['path'] for r in permit['sources']}:
        raise ValueError('Actual probe is not source bound')
    args.job.mkdir()
    started=time.monotonic()
    def check():
        if time.monotonic()>=args.deadline-5:
            raise TimeoutError('GPU probe deadline')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(417)
    # This attempt's explicit allocator ceiling leaves room for CUDA context and
    # other users; it is not a change to formal batch/precision/exposure.
    if permit.get('allocator_cap_bytes'):
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(permit['allocator_cap_bytes']/total,0)
    for row in permit['model_files']: queue.verify(row)
    check()
    settings=replace(policy.baseline(),pooling=args.pooling)
    if args.pooling!='cls':
        settings=replace(settings,encoder_lr=2.5e-5,head_lr=3e-4,head_schedule='encoder',bce_weight=.25,hard_weight=0.)
    model=method.Model(BertModel.from_pretrained(permit['model_path'],local_files_only=True),
        BertTokenizer.from_pretrained(permit['model_path'],local_files_only=True,do_lower_case=True),settings).cuda()
    current,history=handwritten('Agora'),handwritten('Dream')
    teacher=model.teacher(history,check)+.1
    optimizer=method.make_optimizer(model)
    before_head=next(model.head.parameters()).detach().clone()
    before_encoder=next(model.encoder.parameters()).detach().clone()
    intervals=[]
    torch.cuda.reset_peak_memory_stats()
    for update in (1,2):
        check();run.set_rates(optimizer,settings,update,20)
        begin=time.monotonic()
        logs=method.update_h(model,optimizer,current,history,teacher,700+update,check)
        torch.cuda.synchronize()
        intervals.append(dict(begin=begin,end=time.monotonic(),logs=logs))
        queue.write(args.job/'updates.json',dict(pid=os.getpid(),intervals=intervals))
    changed=dict(head=not torch.equal(before_head,next(model.head.parameters())),
                 encoder=not torch.equal(before_encoder,next(model.encoder.parameters())))
    if not all(changed.values()): raise RuntimeError('Actual parameter update missing')
    peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    queue.write(args.job/'result.json',dict(status='HANDWRITTEN_GPU_COMPLETE',pooling=args.pooling,
        updates=2,changed=changed,peak=peak,wall_seconds=time.monotonic()-started,
        records_per_group=32,channel_tokens=256,formal_data_reads=0,intervals=intervals))


if __name__=='__main__':main()
