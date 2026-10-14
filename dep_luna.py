"""Luna 6 subscription decisions. Labels only: no invented probability map.

Imports make no calls and read no credentials. This local-only client uses the
owner's existing Codex login, never an API key or a third-party credential host.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

from deepseek_flash import build_request

MODEL = 'gpt-6-luna'
ENDPOINT = 'https://chatgpt.com/backend-api/codex/responses'
INSTRUCTIONS = 'Classify the supplied context. Return only the requested answer code.'


@dataclass
class SubscriptionClient:
    auth_path: Path = field(default_factory=lambda: Path.home() / '.codex/auth.json', repr=False)

    def post(self, body):
        auth = json.loads(Path(self.auth_path).read_text())
        tokens = auth.get('tokens') or {}
        if auth.get('auth_mode') != 'chatgpt' or not all(
            isinstance(tokens.get(k), str) and tokens[k] for k in ('access_token', 'account_id')
        ):
            raise ValueError('Existing owner Codex subscription login required')
        request = urllib.request.Request(ENDPOINT, json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        request.add_unredirected_header('Authorization', 'Bearer ' + tokens['access_token'])
        request.add_unredirected_header('ChatGPT-Account-Id', tokens['account_id'])
        events = []
        with urllib.request.urlopen(request, timeout=90) as response:
            for line in response:
                if line.startswith(b'data: '):
                    value = line[6:].strip()
                    if value != b'[DONE]':
                        events.append(json.loads(value))
        return events


@dataclass
class Result:
    ok: bool = False
    label: str | None = None
    probs: None = None
    probs_source: str = 'label_only_no_calibrated_distribution'
    model: str = MODEL
    status: int | None = None
    error: str | None = None
    latency_s: float = 0
    usage: dict = field(default_factory=dict)
    raw: list = field(default_factory=list)
    request_body: dict = field(default_factory=dict)


class LunaAdapter:
    def __init__(self, client):
        self.client = client

    def build(self, task):
        old, labels = build_request(task.state, task.question, MODEL)
        rename = {k: k for k in labels}
        if task.question['type'] == 'noul':
            rename = {'true': 'yes', 'false': 'no'}
        elif task.question['type'] == 'score':
            rename = {k: str(int(k) - 1) for k in labels}
        if set(rename.values()) != set(task.labels):
            raise ValueError('Canonical label mismatch')
        body = {'model': MODEL, 'instructions': INSTRUCTIONS,
                'input': [{'role': 'user', 'content': [{'type': 'input_text',
                          'text': old['messages'][0]['content']}]}],
                'reasoning': {'effort': 'none'}, 'store': False, 'stream': True,
                'include': ['message.output_text.logprobs']}
        return body, labels, rename

    def run(self, task):
        result = Result()
        started = time.perf_counter()
        try:
            body, labels, rename = self.build(task)
            result.request_body = body
            result.raw = self.client.post(body)
            result.status = 200
            completed = [e['response'] for e in result.raw if e.get('type') == 'response.completed']
            if len(completed) != 1:
                raise ValueError('Missing unique completed response')
            response = completed[0]
            if response.get('status') != 'completed' or response.get('model') != MODEL:
                raise ValueError('Wrong model or incomplete response')
            usage = response.get('usage') or {}
            reasoning = (usage.get('output_tokens_details') or {}).get('reasoning_tokens')
            result.usage = dict(usage, reasoning_tokens=reasoning)
            if reasoning != 0:
                raise ValueError('Zero reasoning tokens not verified')
            items = [e['item'] for e in result.raw if e.get('type') == 'response.output_item.done']
            if len(items) != 1 or items[0].get('type') != 'message':
                raise ValueError('Expected one answer message, no tools')
            parts = items[0].get('content') or []
            if len(parts) != 1 or parts[0].get('type') != 'output_text':
                raise ValueError('Expected one text answer')
            text = parts[0]['text'].strip().casefold()
            matches = [k for k, mark in labels.items() if text == mark.casefold()]
            if len(matches) != 1:
                raise ValueError('Answer outside requested marks')
            result.label, result.ok = rename[matches[0]], True
        except urllib.error.HTTPError as error:
            result.status, result.error = error.code, 'HTTPError'
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.error = type(error).__name__
        finally:
            result.latency_s = time.perf_counter() - started
        return result
