"""Opt-in GPU-free HTTP regression with the real pinned tokenizer/template."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request


SRC = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get('TF_VISION_TEST_TOKENIZER'),
                     'set TF_VISION_TEST_TOKENIZER to the pinned model tokenizer directory')
class ImageHTTPRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        log_path = Path(cls.temp.name) / 'server.log'
        cls.log = log_path.open('w')
        cls.addClassCleanup(cls.log.close)
        env = os.environ.copy()
        env.update(TF_FAKE_VISION='1',
                   TF_FAKE_TOKENIZER=os.environ['TF_VISION_TEST_TOKENIZER'])
        cls.process = subprocess.Popen([os.environ.get('TF_VISION_TEST_SERVER', str(SRC / 'zig-out/server/fake_serve')), 'serve',
            str(SRC / 'zig/tests/server/fixtures'), '--host', '127.0.0.1', '--port', '0',
            '--name', 'dsv41'], env=env, stdout=cls.log, stderr=subprocess.STDOUT)
        cls.addClassCleanup(cls.close_server)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            match = re.search(r'PORT (\d+)', log_path.read_text())
            if match:
                cls.url = 'http://127.0.0.1:' + match[1]
                return
            if cls.process.poll() is not None:
                raise RuntimeError(log_path.read_text())
            time.sleep(.02)
        raise TimeoutError('fake vision fixture did not announce a port')

    @classmethod
    def close_server(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait()

    def post(self, route, body):
        request = urllib.request.Request(self.url + route, json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read().decode()
                return response.status, raw if body.get('stream') else json.loads(raw)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def body(self, count=1, text='What is shown?'):
        # The fake hook admits these bytes as four rows; pixel decoding and the real tower have separate checks.
        parts = [{'type': 'text', 'text': text}] + [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aA=='}}
            for _ in range(count)]
        return {'model': 'dsv41', 'messages': [{'role': 'user', 'content': parts}],
                'max_tokens': 16, 'temperature': 0,
                'chat_template_kwargs': {'enable_thinking': False}}

    def test_chat_and_token_count_agree_on_image_spans(self):
        for count in (1, 2):
            body = self.body(count)
            status, answer = self.post('/v1/chat/completions', body)
            self.assertEqual(status, 200, answer)
            status, tokens = self.post('/v1/tokenize', body)
            self.assertEqual(status, 200, tokens)
            self.assertEqual(tokens['count'], answer['usage']['prompt_tokens'])
            self.assertEqual(tokens['tokens'].count(129264), count * 4)

    def test_literal_placeholder_with_image_creates_no_extra_span(self):
        status, result = self.post('/v1/tokenize', self.body(text='Literal <｜deepseek_image｜>'))
        self.assertEqual(status, 200, result)
        self.assertEqual(result['tokens'].count(129264), 4)

    def test_text_only_literal_placeholder_remains_refused(self):
        body = self.body(0)
        body['messages'][0]['content'] = 'Literal <｜deepseek_image｜>'
        status, _ = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 400)

    def test_remote_image_urls_remain_refused(self):
        body = self.body()
        body['messages'][0]['content'][1]['image_url']['url'] = 'http://127.0.0.1/private'
        status, _ = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 400)

    def test_streamed_image_reply_finishes(self):
        body = self.body()
        body.update(stream=True, stream_options={'include_usage': True})
        status, result = self.post('/v1/chat/completions', body)
        self.assertEqual(status, 200)
        self.assertIn('data: [DONE]', result)


if __name__ == '__main__':
    unittest.main()
