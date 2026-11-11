"""Exercise authenticated extraction with a tiny checkpoint and independently pinned expected outputs."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import prepare_vision as prepare


def packed(tensors):
    table, chunks, offset = {}, [], 0
    for name, (dtype, raw) in sorted(tensors.items()):
        table[name] = {'dtype': dtype, 'shape': [1], 'data_offsets': [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    header = json.dumps(table, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 8)
    return struct.pack('<Q', len(header)) + header + b''.join(chunks)


def pin(path, data):
    return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vision = self.root / 'vision.safetensors'
        self.bias = self.root / 'bias.safetensors'
        self.tensors = {'vision.test.' + str(i): ('BF16', b'\x00\x3f') for i in range(266)}
        self.tensors.update({'layers.' + str(i) + '.ffn.gate.bias_vl': ('F32', b'\x00\x00\x80\x3f')
                             for i in range(43)})
        self.source = {**self.tensors, 'embed.weight': ('BF16', b'\x00\x00')}
        self.shard_name = 'model-sample.safetensors'
        self.index_name = 'model.safetensors.index.json'
        source = packed(self.source)
        index = json.dumps({'weight_map': {name: self.shard_name for name in self.source}}).encode()
        (self.root / self.shard_name).write_bytes(source)
        (self.root / self.index_name).write_bytes(index)
        biases = {k: v for k, v in self.tensors.items() if k.endswith('.ffn.gate.bias_vl')}
        self.pins = {'repository': 'example/checkpoint', 'revision': '0' * 40,
                     'index': pin(self.index_name, index), 'shards': {self.shard_name: pin(self.shard_name, source)},
                     'outputs': {'vision': pin('vision', packed(self.tensors)), 'bias': pin('bias', packed(biases))}}
        self.addCleanup(patch.stopall)
        patch.object(prepare, 'load_pins', return_value=self.pins).start()

    def run_extract(self):
        return prepare.extract(self.root, self.vision, self.bias)

    def test_extracts_only_vision_and_biases_with_exact_pinned_bytes(self):
        receipt = self.run_extract()
        self.assertEqual(receipt['status'], 'pass')
        self.assertEqual(receipt['vision']['tensors'], 309)
        self.assertEqual(receipt['bias']['tensors'], 43)
        self.assertEqual(self.vision.read_bytes(), packed(self.tensors))
        self.assertNotIn('embed.weight', prepare.header(self.vision)[0])
        self.assertTrue(all(k.endswith('.ffn.gate.bias_vl') for k in prepare.header(self.bias)[0]))
        self.assertNotIn(str(self.root), json.dumps(receipt))

    def test_corrupt_shard_is_rejected_before_outputs(self):
        f = self.root / self.shard_name
        data = bytearray(f.read_bytes())
        data[-1] ^= 1
        f.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.run_extract()
        self.assertFalse(self.vision.exists())
        self.assertFalse(self.bias.exists())

    def test_wrong_output_pin_never_publishes_a_file(self):
        self.pins['outputs']['vision']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'derived vision file'):
            self.run_extract()
        self.assertFalse(self.vision.exists())
        self.assertEqual(list(self.root.glob('.vision-*')), [])

    def test_does_not_replace_an_existing_output(self):
        self.bias.write_bytes(b'keep me')
        with self.assertRaisesRegex(ValueError, 'refusing to replace'):
            self.run_extract()
        self.assertEqual(self.bias.read_bytes(), b'keep me')
        self.assertFalse(self.vision.exists())

    def test_source_cannot_escape_the_model_directory(self):
        for name in ('../outside', '/outside', '..'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'direct child'):
                prepare.checked_source(self.root, name, {'path': name})

    def test_symlink_source_is_rejected(self):
        f = self.root / 'linked'
        f.symlink_to(self.root / self.shard_name)
        with self.assertRaisesRegex(ValueError, 'size/type'):
            prepare.checked_source(self.root, f.name, {**self.pins['shards'][self.shard_name], 'path': f.name})

    def test_invalid_tensor_ranges_fail_before_writing(self):
        table, start = prepare.header(self.root / self.shard_name)
        name = next(k for k in table if k.startswith('vision.'))
        for bounds in ([-1, 1], [4, 2], [0, 1 << 30]):
            table[name]['data_offsets'] = bounds
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, 'byte range'):
                prepare.tensor_spec(self.root, {self.shard_name: (table, start)}, name, self.shard_name)

    def test_identical_output_paths_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'different files'):
            prepare.extract(self.root, self.vision, self.vision)


if __name__ == '__main__':
    unittest.main()
