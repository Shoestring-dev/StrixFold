#!/usr/bin/env python3
"""Extract the pinned checkpoint's CPU vision tower and HIP image routing biases without loading the text model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile


def load_pins():
    return json.loads(Path(__file__).with_name('vision-checkpoint.json').read_text())


def selected(name):
    return name.startswith(('vision.', 'aligner.', 'image_')) or name.endswith('.ffn.gate.bias_vl')


def checked_source(model, name, pin):
    if not name or Path(name).name != name or name in ('.', '..') or pin['path'] != name:
        raise ValueError('checkpoint file must be a direct child of the model directory')
    path = model / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size != pin['bytes']:
        raise ValueError('checkpoint file size/type mismatch: ' + name)
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != pin['sha256']:
            raise ValueError('checkpoint file SHA256 mismatch: ' + name)
    return path


def header(path):
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('truncated safetensors header')
        size, = struct.unpack('<Q', raw)
        if size > 64 << 20:
            raise ValueError('oversized safetensors header')
        data = stream.read(size)
        if len(data) != size:
            raise ValueError('truncated safetensors JSON')
    return json.loads(data), 8 + size


def tensor_spec(model, sources, name, shard):
    table, start = sources[shard]
    spec = table[name]
    begin, end = spec['data_offsets']
    if begin < 0 or end <= begin or start + end > (model / shard).stat().st_size:
        raise ValueError('invalid tensor byte range')
    dtype = 'F32' if name.endswith('.ffn.gate.bias_vl') else 'BF16'
    if spec['dtype'] != dtype:
        raise ValueError('unexpected vision tensor dtype')
    elements = 1
    for dim in spec['shape']:
        if type(dim) is not int or dim <= 0:
            raise ValueError('invalid tensor dimension')
        elements *= dim
    if elements * (4 if dtype == 'F32' else 2) != end - begin:
        raise ValueError('tensor shape does not match byte range')
    return dtype, spec['shape'], start + begin, end - begin


def emit(model, sources, chosen, output, expected):
    table, offset, ranges = {}, 0, []
    for name, shard in sorted(chosen.items()):
        dtype, shape, start, size = tensor_spec(model, sources, name, shard)
        table[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + size]}
        offset += size
        ranges.append((model / shard, start, size))
    raw = json.dumps(table, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix='.vision-')
    try:
        with os.fdopen(descriptor, 'wb') as dest:
            dest.write(struct.pack('<Q', len(raw)))
            dest.write(raw)
            for path, start, size in ranges:
                with path.open('rb') as source:
                    source.seek(start)
                    while size:
                        block = source.read(min(16 << 20, size))
                        if not block:
                            raise ValueError('truncated source tensor')
                        dest.write(block)
                        size -= len(block)
            dest.flush()
            os.fsync(dest.fileno())
        with open(temporary, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if os.stat(temporary).st_size != expected['bytes'] or digest != expected['sha256']:
            raise ValueError('derived vision file failed pinned verification')
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return {'sha256': digest, 'bytes': expected['bytes'], 'tensors': len(chosen)}


def extract(model, output, bias_output):
    if output.exists() or output.is_symlink() or bias_output.exists() or bias_output.is_symlink():
        raise ValueError('refusing to replace an existing vision extraction')
    if output.resolve() == bias_output.resolve():
        raise ValueError('vision and bias outputs must be different files')
    pins = load_pins()
    index_path = checked_source(model, pins['index']['path'], pins['index'])
    index = json.loads(index_path.read_text())['weight_map']
    chosen = {name: shard for name, shard in index.items() if selected(name)}
    biases = {name: shard for name, shard in chosen.items() if name.endswith('.ffn.gate.bias_vl')}
    if len(chosen) != 309 or len(biases) != 43:
        raise ValueError('expected 266 vision tensors and 43 image routing biases')
    sources = {}
    for name in sorted(set(chosen.values())):
        if name not in pins['shards']:
            raise ValueError('checkpoint index names an unpinned shard')
        sources[name] = header(checked_source(model, name, pins['shards'][name]))
    return {'schema': 'strixfold-vision-extraction/1', 'status': 'pass',
            'repository': pins['repository'], 'revision': pins['revision'],
            'vision': emit(model, sources, chosen, output, pins['outputs']['vision']),
            'bias': emit(model, sources, biases, bias_output, pins['outputs']['bias'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--bias-out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.model, args.out, args.bias_out), indent=2))


if __name__ == '__main__':
    main()
