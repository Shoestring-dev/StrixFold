"""Hardware-independent checks for optional HIP vision configuration."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'zig/src/families/dsv41' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisionConfigurationTests(unittest.TestCase):
    def test_workspace_reserve_scales_with_the_image_and_operator_budget(self):
        hip = load('vision_hip')
        small = hip.attention_required_bytes(100, 0)
        self.assertEqual(hip.attention_required_bytes(200, 0), small * 2)
        self.assertEqual(hip.attention_required_bytes(100, 32), small + 32 * 1024**2)
        with self.assertRaises(ValueError):
            hip.attention_required_bytes(100, -1)

    def test_worker_refuses_invalid_configured_limits_before_reading_input(self):
        for setting in ('TF_DS_VISION_THREADS', 'TF_DS_VISION_TIMEOUT'):
            with self.subTest(setting=setting):
                cpu = load('vision_cpu')
                cpu.torch = Mock()
                with patch.dict(os.environ, {setting: '0'}), patch.object(cpu.signal, 'alarm'):
                    with self.assertRaises(ValueError):
                        cpu.main()
                cpu.torch.set_num_threads.assert_not_called()

    def test_worker_timeout_covers_torch_import(self):
        source = ROOT / 'zig/src/families/dsv41/vision_cpu.py'
        code = ("import builtins,time; original=builtins.__import__\n"
                "def slow_import(name,*args,**kwargs):\n"
                " if name=='torch': time.sleep(10)\n"
                " return original(name,*args,**kwargs)\n"
                "builtins.__import__=slow_import\n"
                "exec(compile(open(__import__('sys').argv[1]).read(),'<vision-worker>','exec'),{'__name__':'__main__'})")
        result = subprocess.run([sys.executable, '-c', code, source],
                                env={**os.environ, 'TF_DS_VISION_TIMEOUT': '1'},
                                capture_output=True, timeout=3)
        self.assertEqual(result.returncode, -signal.SIGALRM)

    def test_code_object_build_uses_the_selected_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / 'rocm/bin/amdclang++'
            compiler.parent.mkdir(parents=True)
            compiler.write_text('#!/usr/bin/env python3\nimport json,pathlib,sys\n'
                                'out=pathlib.Path(sys.argv[sys.argv.index("-o")+1])\n'
                                'out.write_bytes(b"test code")\n'
                                'out.with_suffix(".args").write_text(json.dumps(sys.argv[1:]))\n')
            compiler.chmod(0o755)
            output = root / 'worker'
            script = ROOT / 'tools/dsv41-hip/build_vision_worker.sh'
            subprocess.run(['bash', script, root / 'rocm', 'gfx1100', output],
                           check=True, capture_output=True)
            arguments = json.loads((output / 'vision-softmax.args').read_text())
            self.assertIn('--offload-arch=gfx1100', arguments)
            self.assertTrue((output / 'vision-worker.py').is_file())


if __name__ == '__main__':
    unittest.main()
