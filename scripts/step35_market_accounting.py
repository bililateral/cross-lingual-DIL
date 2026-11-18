"""Settle ancillary operations into the one FIT12 ledger, including failed inspections."""
from contextlib import contextmanager
import fcntl
import math
import signal
import time

from step30_evolution_io import Ledger, read, write


# The dispatcher protects a bounded window for serial management operations.
# Completed events consume their measured lifetimes only, including failures.
MARGIN = 60
OP_SECONDS = 25


class MarketLedger(Ledger):
    operation_margin = MARGIN

    def __init__(self,root,prior,cpu_seconds,prior_ids,*,caps):
        self.operation_ids=set(prior_ids)
        self.last_merge=0.
        super().__init__(root,prior,cpu_seconds,caps=caps,cpu_test_cap=1800,tail_seconds=0)
        self.sync_external(force=True)

    def save(self):
        write(self.root/'budget.json',dict(used_seconds=self.used,caps=self.caps,events=self.events,
              cpu_test_seconds=self.cpu_tests,operation_ids=sorted(self.operation_ids),
              management_reserve_seconds=MARGIN,conservative_tail_seconds=0))

    def sync_external(self, force=False):
        if not force and time.monotonic()-self.last_merge<1:
            return
        self.last_merge=time.monotonic()
        for p in sorted((self.root.parent/'evidence').glob('operation_*.json')):
            if p.name in self.operation_ids:
                continue
            row=read(p)
            seconds=row['charged_seconds']
            if type(seconds)!=int or seconds<0:
                raise ValueError('Invalid operation cost')
            self.operation_ids.add(p.name)
            self.events.append(dict(operation=p.name,kind=row['operation'],bucket='auxiliary',
                                    charged_seconds=seconds,success=row.get('success',True)))
            self.used['auxiliary']+=seconds
            self.save()
        if self.used['auxiliary']>self.caps['auxiliary']:
            raise TimeoutError('Total auxiliary budget exceeded')

    def reserve(self,bucket,seconds):
        self.sync_external(force=True)
        super().reserve('auxiliary',MARGIN)
        super().reserve(bucket,seconds+(MARGIN if bucket=='auxiliary' else 0))

    @classmethod
    def settle_closed(cls,root):
        # Only after the owning controller exits, under the exclusive management lock.
        b=read(root/'budget.json')
        obj=cls.__new__(cls)
        obj.root,obj.used,obj.events,obj.caps=root,b['used_seconds'],b['events'],b['caps']
        obj.cpu_tests,obj.operation_ids,obj.last_merge=b['cpu_test_seconds'],set(b['operation_ids']),0.
        obj.sync_external(force=True)


@contextmanager
def billed_operation(root, operation, settle_closed=lambda:None):
    """Unacknowledged charges and the next reservation must fit in the protected window."""
    evidence=root/'evidence'
    evidence.mkdir(exist_ok=True)
    with (evidence/'operations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        snapshot=root/'campaign/budget.json'
        if snapshot.exists():
            budget=read(snapshot)
            booked=set(budget['operation_ids'])
            unbooked=sum(read(p)['charged_seconds'] for p in evidence.glob('operation_*.json') if p.name not in booked)
            if (unbooked+OP_SECONDS>MARGIN or
                    budget['used_seconds']['auxiliary']+unbooked+OP_SECONDS>budget['caps']['auxiliary']):
                raise TimeoutError('Waiting for settled management headroom')
        started=time.monotonic()
        dest=evidence/f'operation_{time.time_ns()}.json'
        success=False
        def deadline(signum,frame):
            raise TimeoutError('Management operation lifecycle deadline')
        before=signal.signal(signal.SIGALRM,deadline)
        signal.setitimer(signal.ITIMER_REAL,OP_SECONDS-2)
        try:
            yield
            success=True
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,before)
            elapsed=time.monotonic()-started
            write(dest,dict(operation=operation,success=success,actual_seconds=elapsed,
                            charged_seconds=math.ceil(elapsed)))
            settle_closed()
            if elapsed>OP_SECONDS:
                raise TimeoutError('Management operation exceeded its reservation')
