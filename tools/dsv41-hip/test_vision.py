"""Offline trust-boundary checks for the optional vision preparation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import sys
import unittest
from unittest.mock import patch

import prepare_vision as extract


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'zig/src/families/dsv41/vision_cpu.py'


@unittest.skipUnless(SOURCE.is_file(), 'optional vision source checkout absent')
class VisionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('optional_cpu_vision', SOURCE)
        cls.cpu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cpu)

    def test_grid_sizes_and_span_count(self):
        for h, w in ((3, 3), (4, 7), (39, 39), (90, 90)):
            lh, lw = (h + 2) // 3, (w + 2) // 3
            header = struct.pack('<8I', h * 14, w * 14, h, w, lh, lw, 0, 0)
            self.assertEqual(self.cpu.decode_grid(header), (h, w, h * w, lh * (lw + 1) + 2))

    def test_rejects_bad_or_unbounded_grids_before_tensor_allocation(self):
        cases = (b'', b'X' * 31,
                 struct.pack('<8I', 0, 0, 0, 0, 0, 0, 0, 0),
                 struct.pack('<8I', 14, 14, 999999, 999999, 1, 1, 0, 0),
                 struct.pack('<8I', 42, 42, 3, 3, 2, 2, 0, 0),
                 struct.pack('<8I', 1400, 1400, 100, 100, 34, 34, 0, 0))
        for header in cases:
            with self.subTest(header=header), self.assertRaises(ValueError):
                self.cpu.decode_grid(header)

    def test_weights_require_digest_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'weights'
            p.write_bytes(b'untrusted weights')
            for digest in ('', 'a' * 64):
                with self.assertRaises(ValueError):
                    self.cpu.read_weights(p, digest)

    def test_weights_reject_oversized_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'weights'
            p.write_bytes(struct.pack('<Q', (1 << 20) + 1))
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'oversized'):
                self.cpu.read_weights(p, digest)


class ExtractionBoundaryTests(unittest.TestCase):
    def test_selected_tensors_exclude_text_weights(self):
        for name in ('vision.blocks.1.norm1.weight', 'aligner.w1.weight', 'image_start',
                     'layers.0.ffn.gate.bias_vl'):
            self.assertTrue(extract.selected(name))
        for name in ('layers.0.ffn.gate.weight', 'embed.weight', 'head.weight'):
            self.assertFalse(extract.selected(name))

    def test_untrusted_index_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            index = p / 'model.safetensors.index.json'
            index.write_bytes(b'evil')
            pins = {'index': {'path': index.name, 'bytes': 4, 'sha256': '0' * 64}}
            with patch.object(extract, 'load_pins', return_value=pins):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    extract.extract(p, p / 'vision', p / 'bias')
            self.assertFalse((p / 'vision').exists())
            self.assertFalse((p / 'bias').exists())

    def test_truncated_and_oversized_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'weights'
            for content in (b'x', struct.pack('<Q', 65 << 20), struct.pack('<Q', 20) + b'{}'):
                p.write_bytes(content)
                with self.assertRaises(ValueError):
                    extract.header(p)


@unittest.skipUnless(SOURCE.is_file(), 'optional vision source checkout absent')
class VisionAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SOURCE.parents[4] / 'tools/dsv41-hip/verify_vision.py'
        spec = importlib.util.spec_from_file_location('optional_vision_admission', path)
        cls.admission = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.admission)

    def test_authenticates_bias_without_starting_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bias'
            p.write_bytes(b'pinned bias')
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(self.admission.subprocess, 'run') as run:
                self.admission.verify(p, digest)
                run.assert_not_called()

    def test_authenticates_encoder_weights_and_pins_python_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'weights'
            p.write_bytes(b'pinned extraction')
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(self.admission.subprocess, 'run') as run:
                self.admission.verify(p, digest, sys.executable, p, digest)
                argv = run.call_args.args[0]
                self.assertEqual(argv[:3], [sys.executable, '-I', '-c'])
                self.assertIn('2.9.0+cpu', argv[3])
                self.assertTrue(run.call_args.kwargs['check'])
                self.assertEqual(run.call_args.kwargs['timeout'], 30)

    def test_rejects_symlink_and_invalid_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'weights'
            p.write_bytes(b'pinned extraction')
            link = Path(tmp) / 'link'
            link.symlink_to(p)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            for path, sha in ((link, digest), (p, ''), (p, '0' * 64)):
                with self.subTest(path=path, sha=sha), self.assertRaises(ValueError):
                    self.admission.checked_file(path, sha)


if __name__ == '__main__':
    unittest.main()
