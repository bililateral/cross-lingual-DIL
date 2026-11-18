"""Market routing and the bounded historical state for the full H FIT12 scorer."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from zipfile import ZipFile, ZipInfo

import torch
from torch import nn

from step29_vendorlink_methods import Memory
from step35_vendorlink_account_io import pinned
from step35_vendorlink_account_policy import DOMAINS


FORMAT = 'fit12-h-market-qv-r8-v1'
CAP = 1 << 20
RANK = 8
LAYERS = (8, 9, 10, 11)


class LowRank(nn.Module):
    def __init__(self, reference):
        super().__init__()
        options = dict(bias=False, device=reference.weight.device, dtype=torch.float32)
        self.down = nn.Linear(reference.in_features, RANK, **options)
        self.up = nn.Linear(RANK, reference.out_features, **options)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


class MarketRoute:
    """Owned by one model. CUDA checkpoint recomputation can run on another Python thread."""
    def __init__(self):
        self.market = None

    def get(self):
        if self.market is None:
            raise RuntimeError('A market route is required for this model call')
        return self.market


class RoutedProjection(nn.Module):
    def __init__(self, base, route):
        super().__init__()
        self.base = base
        self.route = route
        self.banks = nn.ModuleDict()

    def forward(self, x):
        market = self.route.get()
        y = self.base(x)
        return y if market == DOMAINS[0] else y + self.banks[market](x)


class MarketH(nn.Module):
    """One original H, its single relation head, and up to two distinct adapter banks."""
    def __init__(self, original):
        super().__init__()
        if (original.settings.method not in ('H_SEQ', 'H_ER', 'H_full')
                or original.encoder.config.hidden_size != 768
                or original.encoder.config.num_hidden_layers != 12):
            raise ValueError('Expected FIT12 BERT-base H')
        self.h = original
        self.stage = 0
        self._active = MarketRoute()
        self.projections = {}
        for i in LAYERS:
            attention = self.h.encoder.encoder.layer[i].attention.self
            for name in ('query', 'value'):
                base = getattr(attention, name)
                if not isinstance(base, nn.Linear) or base.weight.shape != (768, 768):
                    raise ValueError('Expected the original query/value projection')
                wrapped = RoutedProjection(base, self._active)
                setattr(attention, name, wrapped)
                self.projections[f'{i}.{name}'] = wrapped  # Registered by the original encoder.

    @property
    def head(self):
        return self.h.head

    @property
    def settings(self):
        return self.h.settings

    @property
    def encoder(self):
        return self.h.encoder

    @property
    def auxiliary_optimizer(self):
        return self.h.auxiliary_optimizer

    def start_account_stage(self, classes, seed):
        self.h.start_account_stage(classes, seed)

    def backward_h(self, group, step_seed, teacher=None, history=False, check=lambda: None):
        if history != (DOMAINS.index(group.domain) < self.stage-1) or DOMAINS.index(group.domain) >= self.stage:
            raise ValueError('Training group must use its current or retained origin market')
        with self.routing(group.domain):
            return self.h.backward_h(group, step_seed, teacher, history, check, frozen_history=history)

    def backward_birth(self, group, teacher, check):
        if DOMAINS.index(group.domain) >= self.stage-1:
            raise ValueError('Birth targets belong to completed markets')
        with self.routing(group.domain):
            return self.h.backward_birth(group, teacher, check, frozen_history=True)

    def shared_encoder_state(self):
        if self.stage != 1:
            raise ValueError('Save the single shared backbone when leaving AlphaBay')
        return {key.replace('.query.base.', '.query.').replace('.value.base.', '.value.'): value.detach().cpu()
                for key, value in self.encoder.state_dict().items()}

    def start_training_market(self, stage, initialization_seed, optimizer):
        from step29_vendorlink_methods import make_optimizer
        device = next(self.encoder.parameters()).device
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            torch.manual_seed(initialization_seed)
            self.enter(stage)
        if stage == 1:
            return make_optimizer(self.h)
        # Keep the single head and its Adam moments, discard the previous market's moments.
        enc_group = optimizer.param_groups[0]
        for param in enc_group['params']:
            param.grad = None
            optimizer.state.pop(param, None)
        enc_group['params'] = [p for p in self.encoder.parameters() if p.requires_grad]
        if len(enc_group['params']) != 2*len(self.projections):
            raise ValueError('Only the current query/value factors may train in the encoder')
        return optimizer

    def enter(self, stage: int) -> None:
        if stage != self.stage + 1 or stage not in (1, 2, 3):
            raise ValueError('Enter the next market in chronological order')
        if stage == 2:
            self.h.encoder.requires_grad_(False)
            for p in self.projections.values():
                p.banks['Dream'] = LowRank(p.base)
        if stage == 3:
            for p in self.projections.values():
                p.banks['Agora'] = deepcopy(p.banks['Dream'])
                p.banks['Dream'].requires_grad_(False)
        self.stage = stage

    @contextmanager
    def routing(self, market):
        # A future market uses the last available encoder for its pre-learning reading.
        market_index = DOMAINS.index(market)
        route = DOMAINS[min(market_index, max(0, self.stage - 1))]
        # Each worker calls this model serially. Backward joins its autograd workers
        # before leaving the scope, including recomputation performed off this thread.
        previous = self._active.market
        self._active.market = route
        try:
            yield
        finally:
            self._active.market = previous

    def records(self, group, check):
        with self.routing(group.domain):
            return self.h.records(group, check)

    def teacher(self, group, check):
        with self.routing(group.domain):
            return self.h.teacher(group, check)

    def adapter_tensors(self) -> dict:
        return {f'{site}.{market}.{key}': parameter
                for site, projection in self.projections.items()
                for market, bank in projection.banks.items()
                for key, parameter in bank.named_parameters()}

    @torch.no_grad()
    def load_adapters(self, state: dict) -> None:
        targets = self.adapter_tensors()
        if state.keys() != targets.keys():
            raise ValueError('Adapter sites or market banks differ from this stage')
        for key, target in targets.items():
            value = state[key]
            if value.shape != target.shape or value.dtype != torch.float32:
                raise ValueError(f'Invalid FP32 adapter tensor {key}')
            target.copy_(value)


def metadata(stage: int, identity: str) -> dict:
    return dict(format=FORMAT, identity=identity, stage=stage, markets=list(DOMAINS),
                layers=list(LAYERS), projections=['query', 'value'], rank=RANK, scale=1,
                base='single_shared_encoder', head='single_live_relation_head')


def pack_history(model: MarketH, memory: Memory, identity: str) -> bytes:
    replay = model.settings.method in ('H_ER', 'H_full')
    if model.stage not in (1, 2, 3) or len(memory.entries) != (6 if replay else 0):
        raise ValueError('Market stage and method-specific retained group count must match')
    adapter_state = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.adapter_tensors().items()}, adapter_state)
    target = io.BytesIO()
    with ZipFile(target, 'w') as archive:
        # Fixed member names and uncompressed payloads give byte-exact additive accounting.
        archive.writestr(ZipInfo('route.json'), json.dumps(metadata(model.stage, identity),
                          separators=(',', ':')).encode('utf-8'))
        archive.writestr(ZipInfo('adapters.pt'), adapter_state.getvalue())
        if replay:
            archive.writestr(ZipInfo('memory.pt'), memory.bytes())
    payload = target.getvalue()
    if len(payload) > min(model.settings.memory_bytes, CAP):
        raise ValueError(f'Historical state exceeds the 1 MiB cap ({len(payload)} bytes)')
    return payload


def restore_history(model: MarketH, payload: bytes, identity: str, memory=None) -> None:
    if len(payload) > min(model.settings.memory_bytes, CAP):
        raise ValueError('Historical state exceeds the 1 MiB cap')
    with ZipFile(io.BytesIO(payload)) as archive:
        replay = model.settings.method in ('H_ER', 'H_full')
        expected = ['route.json', 'adapters.pt'] + (['memory.pt'] if replay else [])
        if archive.namelist() != expected:
            raise ValueError('Unexpected members for this market history method')
        if json.loads(archive.read('route.json')) != metadata(model.stage, identity):
            raise ValueError('Historical state identity or routing differs')
        state = torch.load(io.BytesIO(archive.read('adapters.pt')), map_location='cpu', weights_only=True)
        model.load_adapters(state)
        if memory is not None and replay:
            state = torch.load(io.BytesIO(archive.read('memory.pt')), map_location='cpu', weights_only=True)
            memory.restore(state)
            if len(memory.entries) != 6:
                raise ValueError('Expected six restored replay groups')
        elif memory is not None and memory.entries:
            raise ValueError('SEQ must not retain replay groups')


def restore_scoring_endpoint(original, value: dict, identity: str, stage: int) -> MarketH:
    if value['task']['market_adapters'] != FORMAT or stage not in (0, 1, 2, 3):
        raise ValueError('Unknown routed H endpoint')
    if stage:
        base = torch.load(pinned(value['shared_encoder']), map_location='cpu', weights_only=True)
        original.encoder.load_state_dict(base, strict=True)
        del base
    original.head.load_state_dict(value['head'], strict=True)
    model = MarketH(original)
    for index in range(1, stage + 1):
        model.enter(index)
    if stage >= 2 or 'market_state' in value:
        restore_history(model, pinned(value['market_state']).read_bytes(), identity)
    model.eval()
    return model
