"""Pinned official files only; file integrity/header inspection, no model loading."""
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request

ROOT = Path('/home/yongpeng/cross-lingual')
OUT = ROOT/'reports/documentation/20261029/vendorlink/data_qualification'
meta = json.loads((OUT/'hf_model_api.json').read_text())
revision = meta['sha']
target = ROOT/'models/step29/bert_base_cased'
target.mkdir(parents=True, exist_ok=False)
names = ['config.json', 'tokenizer_config.json', 'vocab.txt', 'tokenizer.json', 'README.md', 'model.safetensors']
started = time.monotonic()
receipt = {'repository': 'google-bert/bert-base-cased', 'revision': revision,
           'target': str(target), 'download_host': 'Linux', 'windows_weights_copy': False,
           'model_loaded': False, 'started_server': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': []}
try:
    entries = {x['rfilename']: x for x in meta['siblings']}
    for name in names:
        entry = entries[name]
        url = f'https://huggingface.co/google-bert/bert-base-cased/resolve/{revision}/{name}'
        pending = target/(name+'.partial')
        sha = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(url, timeout=45) as source, pending.open('xb') as dest:
            while block := source.read(1024*1024):
                if time.monotonic()-started > 1200:
                    raise TimeoutError('Download wall limit 1200 seconds')
                dest.write(block)
                sha.update(block)
                count += len(block)
        assert count == entry['size'], (name, count, entry['size'])
        if 'lfs' in entry:
            assert sha.hexdigest() == entry['lfs']['sha256'], name
        else:
            blob = hashlib.sha1(b'blob '+str(count).encode()+b'\0'+pending.read_bytes()).hexdigest()
            assert blob == entry['blobId'], name
        pending.rename(target/name)
        receipt['files'].append({'name': name, 'bytes': count, 'sha256': sha.hexdigest(),
                                  'official_blob_id': entry['blobId'], 'integrity_verified': True})
        print('VERIFIED', name, count, flush=True)
    config = json.loads((target/'config.json').read_text())
    with (target/'model.safetensors').open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(size))
    receipt['config'] = {k: config.get(k) for k in ('model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size', 'max_position_embeddings')}
    receipt['tensor_count'] = len([k for k in header if k != '__metadata__'])
    receipt['pooler_keys'] = [k for k in header if 'pooler' in k]
    receipt['embedding_shape'] = header['bert.embeddings.word_embeddings.weight']['shape']
    assert receipt['pooler_keys'], 'VendorLink needs pretrained pooler; do not silently initialize it'
    assert receipt['config']['model_type'] == 'bert' and receipt['config']['hidden_size'] == 768
    receipt['status'] = 'FILES_VERIFIED_NOT_MODEL_LOAD_QUALIFIED'
except BaseException as exc:
    receipt['status'] = 'DOWNLOAD_OR_FILE_CHECK_FAILED'
    receipt['error'] = repr(exc)
    raise
finally:
    receipt['wall_seconds'] = time.monotonic()-started
    receipt['finished_server'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (OUT/'model_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(receipt['status'], flush=True)
