"""One-question state, offline tokenization, no SDK retries or other network calls."""
import json
import os
import re
import urllib.error
import urllib.request
import shutil
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
from common import config_path, sha256


def load_binding(config):
    path = config_path(config, 'runtime_binding')
    if not path.exists():
        raise ValueError('No runtime_binding.json: bind the official runtime and API request first')
    b = json.loads(path.read_text(encoding='utf-8'))
    if b.get('model_repo') != 'Qwen/Qwen3.5-4B':
        raise ValueError('Only Qwen/Qwen3.5-4B is allowed')
    if not re.fullmatch('[a-f0-9]{40}', b.get('model_revision', '')):
        raise ValueError('Official model revision must be an immutable 40-character commit')
    runtime = Path(b['official_runtime_path'])
    if not runtime.is_absolute():
        runtime = path.parent/runtime
    if runtime.exists() and sha256(runtime) != b['official_runtime_sha256']:
        raise ValueError('runtime.json changed; review and bind its official parameters again')
    if not re.fullmatch('[a-f0-9]{64}', b.get('official_runtime_sha256', '')):
        raise ValueError('Missing official runtime provenance hash')
    body = b['request_body']
    forbidden = {'model', 'messages', 'stream', 'n', 'tools', 'extra_body'}
    if forbidden & set(body):
        raise ValueError('request_body must contain only generation settings, raw HTTP format')
    fields = [k for k in ('max_tokens', 'max_completion_tokens') if k in body]
    if len(fields) != 1 or type(body[fields[0]]) is not int:
        raise ValueError('Exactly one official output-limit parameter is required')
    b['output_cap'] = body[fields[0]]
    if not 1 <= b['output_cap'] <= config['limits']['total_output']:
        raise ValueError('Official output cap incompatible with task limits; do not silently modify it')
    if b.get('completion_budget_includes_reasoning') is not True:
        raise ValueError('Endpoint must limit and report ALL output, including thinking')
    if not isinstance(b.get('timeout_seconds'), (int, float)) or b['timeout_seconds'] <= 0:
        raise ValueError('Official positive request timeout is required')
    return b


def prepare_tokenizer_assets(config, binding, destination):
    """Build-time only. All prediction-time assets reside within --index."""
    from common import dump
    source = config_path(config, 'tokenizer_dir')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (source/'manifest.json').exists():
        # Validate pinned local files before reusing them.
        TokenCounter(config, binding, source)
        for filename in ('tokenizer.json', 'chat_template.jinja', 'manifest.json'):
            if (source/filename).resolve() != (destination/filename).resolve():
                shutil.copyfile(source/filename, destination/filename)
    else:
        hashes = {}
        for filename in ('tokenizer.json', 'chat_template.jinja'):
            url = f"https://huggingface.co/Qwen/Qwen3.5-4B/resolve/{binding['model_revision']}/{filename}"
            with urllib.request.urlopen(url, timeout=binding['timeout_seconds']) as response:
                (destination/filename).write_bytes(response.read())
            hashes[filename] = sha256(destination/filename)
        dump(destination/'manifest.json', {'repo': 'Qwen/Qwen3.5-4B',
            'revision': binding['model_revision'], 'sha256': hashes})
    TokenCounter(config, binding, destination)


class TokenCounter:
    def __init__(self, config, binding, root=None):
        root = Path(root) if root is not None else config_path(config, 'tokenizer_dir')
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['revision'] != binding['model_revision']:
            raise ValueError('Tokenizer revision does not match official runtime model revision')
        for file, expected in manifest['sha256'].items():
            if sha256(root/file) != expected:
                raise ValueError(f'Tokenizer file changed: {file}')
        self.tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        self.template = env.from_string((root/'chat_template.jinja').read_text(encoding='utf-8'))
        self.kwargs = binding['request_body'].get('chat_template_kwargs', {}).copy()
        # Some endpoints use top-level enable_thinking. The binding documents this explicitly.
        if 'enable_thinking' in binding['request_body']:
            self.kwargs['enable_thinking'] = binding['request_body']['enable_thinking']

    def count(self, messages):
        rendered = self.render(messages)
        return len(self.tokenizer.encode(rendered, add_special_tokens=False).ids)

    def render(self, messages):
        return self.template.render(messages=messages, add_generation_prompt=True,
                                    tools=None, **self.kwargs)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Redirect refused', headers, fp)


class Budget:
    def __init__(self, config, binding, counter, question_id):
        self.config, self.binding, self.counter = config, binding, counter
        self.id = question_id
        self.calls = self.input = self.output = 0
        self.verifiable = True
        self.events = []
        self.opener = urllib.request.build_opener(NoRedirect())

    def fits(self, messages):
        l = self.config['limits']
        p = self.counter.count(messages)+l['prompt_safety_tokens']
        o = self.binding['output_cap']
        return (self.verifiable and self.calls < l['max_calls'] and
                self.input+p <= l['total_input'] and
                self.output+o <= l['total_output'] and p+o <= l['request_total'])

    def call(self, messages, stage):
        if not self.fits(messages):
            raise ValueError('Budget does not permit this request with its full official output cap')
        url = os.environ['LLM_BASE_URL'].rstrip('/')+'/chat/completions'
        if not url.startswith(('http://', 'https://')):
            raise ValueError('LLM_BASE_URL must be an OpenAI-compatible http(s) base URL')
        submission = os.environ['SUBMISSION_ID']
        model = os.environ['LLM_MODEL']
        api_key = os.environ['LLM_API_KEY']
        # A serving alias may differ from repo ID; the official binding records it.
        expected_alias = self.binding.get('served_model')
        if self.binding.get('enforce_served_alias', False) and expected_alias and model != expected_alias:
            raise ValueError('LLM_MODEL differs from the official served-model alias')
        body = dict(self.binding['request_body'], model=model, messages=messages, stream=False)
        request = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+api_key,
                     'X-Question-ID': self.id, 'X-Submission-ID': submission})
        local_prompt = self.counter.count(messages)
        self.calls += 1  # Count attempts, including HTTP errors and ambiguous timeouts.
        event = {'stage': stage, 'local_prompt_tokens': local_prompt,
                 'requested_output_cap': self.binding['output_cap']}
        self.events.append(event)
        try:
            from request_journal import record_attempt
            record_attempt({'id': self.id, 'stage': stage, 'local_prompt_tokens': local_prompt,
                            'requested_output_cap': self.binding['output_cap']})
            with self.opener.open(request, timeout=self.binding['timeout_seconds']) as response:
                data = json.load(response)
            usage = data.get('usage', {})
            prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
            if any(type(v) is not int or v < 0 for v in (prompt, completion)):
                raise ValueError('Missing trustworthy prompt/completion usage')
            self.input += prompt
            self.output += completion
            event.update(input_tokens=prompt, output_tokens=completion)
            l = self.config['limits']
            if (prompt+self.binding['output_cap'] > l['request_total'] or
                    prompt > local_prompt+l['prompt_safety_tokens'] or
                    completion > self.binding['output_cap'] or
                    self.input > l['total_input'] or self.output > l['total_output']):
                raise ValueError('Endpoint template/usage differs from binding or exceeds budget')
            choice = data['choices'][0]
            event['finish_reason'] = choice.get('finish_reason')
            content = choice['message'].get('content')
            if not isinstance(content, str):
                raise ValueError('Missing textual model response')
            return content
        except Exception as e:
            self.verifiable = False
            # No body, credentials, chain of thought, or raw URL in error logs.
            event['error_type'] = type(e).__name__
            raise RuntimeError('Endpoint request failed or usage was not verifiable; see trace') from None
