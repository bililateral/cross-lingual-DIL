# 未更动的 H 和 Memory 源码节录
原模块 step29_vendorlink_methods.py；完整文件的 SHA-256 为 b6b5ce15248719080005f5f29c928544b0f48b59c463a62d1289b1dac613b116。节录范围为 56–70，164–173，183–206，226–230，286 至末行。

~~~python
def text_pool(output, mask, pooling):
    if pooling == 'cls':
        return output.last_hidden_state[:, 0].float()
    if pooling == 'mean_last':
        values = output.last_hidden_state.float()
    elif pooling == 'mean_last4_sum':
        if output.hidden_states is None or len(output.hidden_states) < 5:
            raise ValueError('Four Transformer layer outputs required, excluding embeddings')
        values = torch.stack(output.hidden_states[-4:]).sum(0).float()
    else:
        raise ValueError('Unknown text pooling')
    if mask.shape != values.shape[:2] or (mask.sum(1) == 0).any():
        raise ValueError('Aligned nonempty attention mask required')
    return (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

class Model(nn.Module):
    def __init__(self, bert, tokenizer, settings: Settings):
        super().__init__()
        self.encoder, self.tokenizer, self.settings = bert, tokenizer, settings
        self.head = (nn.Identity() if settings.method == "VendorLink_SEQ" else
                     nn.Sequential(nn.Linear(4*bert.config.hidden_size, 128), nn.ReLU(), nn.Linear(128, 1)))
        self.classifier = None
        self.classes = ()
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)

    def encode(self, texts, limit, hidden=False):
        device = next(self.parameters()).device
        features = self.tokenizer(texts, padding=True, truncation=True, max_length=limit, return_tensors="pt")
        features = {k: v.to(device) for k, v in features.items()}
        output = self.encoder(**features, output_hidden_states=hidden, return_dict=True)
        return output, features["attention_mask"]

    def records(self, group, check=lambda: None):
        records = [r for v in group.views for r in v.records]
        texts = [r.title for r in records]+[r.description for r in records]
        parts = []
        for start in range(0, len(texts), self.settings.microbatch):
            check()
            output, mask = self.encode(texts[start:start+self.settings.microbatch], self.settings.channel_tokens,
                                       hidden=self.settings.pooling == 'mean_last4_sum')
            parts.append(text_pool(output, mask, self.settings.pooling))
        title, description = F.normalize(torch.cat(parts), dim=1).chunk(2)
        return torch.cat((title, description), dim=1)/math.sqrt(2)

    def table(self, z):
        i, j = torch.triu_indices(len(z), len(z), 1, device=z.device)
        return torch.cat([self.head(torch.cat(((z[i[k:k+512]]-z[j[k:k+512]]).abs(),
            z[i[k:k+512]]*z[j[k:k+512]]), 1)).flatten() for k in range(0, len(i), 512)])

    @torch.no_grad()
    def teacher(self, group, check=lambda: None):
        self.eval()
        return self.table(self.records(group, check)).float().cpu().clone()

class Memory:
    def __init__(self, seed, settings):
        self.rng = random.Random(seed)
        self.settings = settings
        self.seen, self.entries = 0, []

    def retain(self, groups, stage, model, check=lambda: None, *, record_teacher=lambda group: None):
        if stage not in (1, 2):
            raise ValueError("Only completed nonfinal stages produce history")
        # Select first, then encode new survivors only; existing teachers never refresh.
        selected = list(self.entries)
        for group in sorted(groups, key=lambda g: g.uid):
            self.seen += 1
            slot = len(selected) if len(selected) < self.settings.memory_groups else self.rng.randrange(self.seen)
            if slot < self.settings.memory_groups:
                row = {"group": asdict(group), "origin": stage, "teacher": None}
                if slot == len(selected):
                    selected.append(row)
                else:
                    selected[slot] = row
        for row in selected:
            if row["teacher"] is None:
                group = Group.from_dict(row["group"])
                row["teacher"] = model.teacher(group, check)
                record_teacher(group)
        self.entries = selected
        if len(self.bytes()) > self.settings.memory_bytes:
            raise ValueError("Retained text, teachers and sampler state exceed the memory budget")

    def state(self):
        return {"seen": self.seen, "entries": self.entries, "rng": self.rng.getstate()}

    def restore(self, state):
        self.seen, self.entries = state["seen"], state["entries"]
        self.rng.setstate(state["rng"])

    def bytes(self):
        buffer = io.BytesIO()
        torch.save(self.state(), buffer)
        return buffer.getvalue()

    def draw(self, draw_seed):
        row = random.Random(draw_seed).choice(self.entries)
        return Group.from_dict(row["group"]), row["teacher"]
~~~

公开容量来源为 20261118/vendorlink_fit_regroup/freeze.json 的 indices。进入 Agora 前，seed 51、52、53 原始字节数依次为 129173、127253、125589。
