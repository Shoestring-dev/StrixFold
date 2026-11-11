"""Vision settings survive the standard launcher's empty-environment boundary."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent


class VisionLauncherTests(unittest.TestCase):
    def launch(self, modality, exported=False, role=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kit = root / 'kit'
            kit.mkdir()
            for name in ('plain-cos', 'plain-sin', 'compressed-cos', 'compressed-sin'):
                (kit / ('rope-' + name + '.f32')).write_bytes(bytes(65 * 128))
            (root / 'DEPLOY.json').write_text(json.dumps({'commit': '0' * 40}))
            config = root / 'serve.conf'
            settings = {'MODALITIES': modality, 'CONTEXT': '16', 'GRAPHS': '0',
                        'DEPLOY_DIR': str(root), 'DS_KIT': str(kit), 'ENGINE_CPUS': '',
                        'LOCK_WAIT_S': '3', 'STREAMS': '1', 'PARALLEL': '1'}
            vision = {'PYTHON': sys.executable, 'WEIGHTS': str(root / 'vision.safetensors'),
                      'SHA256': 'a' * 64, 'HIP_CODE': str(root / 'vision.co'), 'HIP_SHA256': 'b' * 64,
                      'CACHE_MIB': '32', 'THREADS': '2', 'TIMEOUT': '7', 'RESERVE_MIB': '128'}
            if role != 0:
                vision = {}
            if not exported:
                settings.update({'VISION_' + key: value for key, value in vision.items()})
            config.write_text((TOOLS / 'serve.conf').read_text() + '\n' +
                              '\n'.join(key + '=' + shlex.quote(value) for key, value in settings.items()) + '\n')
            code = ('source "$1"; serve_load "$4"; serve_env; '
                    'SERVE_CMD=("$2" -c \'import os,json; print(json.dumps(dict(os.environ)))\'); '
                    'serve_spawn "$3/pid" "$3/output" 1')
            env = {**os.environ, 'TF_SERVE_CONF': str(config)}
            if exported:
                env.update({'TF_DS_VISION_' + key: value for key, value in vision.items()})
            result = subprocess.run(['bash', '-e', '-c', code, 'vision-launch-test',
                                     TOOLS / 'serve_lib.sh', sys.executable, root, str(role)],
                                    env=env, capture_output=True, text=True, timeout=8)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((root / 'output').read_text())
            return data, vision

    def test_configured_image_settings_reach_the_spawned_process(self):
        data, expected = self.launch('text_image')
        for key, value in expected.items():
            self.assertEqual(data.get('TF_DS_VISION_' + key), value, key)
        self.assertEqual(data['PATH'].split(':')[0], str(Path(expected['PYTHON']).parent))

    def test_exported_image_settings_are_preserved_when_config_is_empty(self):
        data, expected = self.launch('text_image', exported=True)
        for key, value in expected.items():
            self.assertEqual(data.get('TF_DS_VISION_' + key), value, key)

    def test_follower_image_runtime_needs_no_encoder_environment(self):
        data, _ = self.launch('text_image', role=1)
        self.assertFalse(any(key.startswith('TF_DS_VISION_') for key in data))

    def test_text_serving_does_not_start_the_optional_image_path(self):
        data, _ = self.launch('text')
        self.assertFalse(any(key.startswith('TF_DS_VISION_') for key in data))


if __name__ == '__main__':
    unittest.main()
