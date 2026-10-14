"""Owner-paid Luna Decisions API; native distributions, no generated JSON/probabilities."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import shlex
import time
import urllib.request
import urllib.error
from deepseek_flash import BenchmarkResult

MODEL = 'gpt-6-luna'
ENDPOINT = 'https://api.openai.com/v1/decisions'


def read_key(path):
    found = []
    for line in Path(path).read_text().splitlines():
        match = re.match(r'^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$', line)
        if match:
            parts = shlex.split(match[1], comments=True)
            if len(parts) != 1:
                raise ValueError('Malformed selected key')
            found.append(parts[0])
    if len(found) != 1 or not found[0]:
        raise ValueError('Missing or duplicated personal OPENAI_API_KEY')
    return found[0]


@dataclass
class DecisionsClient:
    key: str = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError('Personal OPENAI_API_KEY required')

    def post(self, body):
        request = urllib.request.Request(ENDPOINT, json.dumps(body, ensure_ascii=False).encode(),
                                        headers={'Content-Type': 'application/json'})
        request.add_unredirected_header('Authorization', 'Bearer ' + self.key)
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)


class DecisionsAdapter:
    def __init__(self, client):
        self.client = client

    def build(self, task):
        source = task.question
        question = {'name': 'decision', 'instructions': source['instructions']}
        criteria = source.get('criteria')
        kind = source['type']
        if kind == 'noul':
            question['type'] = 'predicate'
            if criteria:
                question['instructions'] += f"\ntrue means: {criteria['true']}\nfalse means: {criteria['false']}"
        elif kind == 'choice':
            question['type'] = 'choice'
            if isinstance(criteria, list): criteria = {k: k for k in criteria}
            question['choices'] = [{'value': k, 'description': str(v)} for k, v in criteria.items()]
        elif kind == 'score':
            question['type'] = 'score'
            question['levels'] = [{'label': str(i), 'description': str(v)} for i, v in enumerate(criteria)]
        else:
            raise ValueError('Unknown task type')
        state = task.state if isinstance(task.state, str) else json.dumps(task.state, ensure_ascii=False, allow_nan=False)
        return {'model': MODEL, 'input': state, 'questions': [question]}

    def decode(self, response, task):
        if response.get('model') != MODEL:
            raise ValueError('Wrong returned model')
        answers = response.get('answers') or []
        if len(answers) != 1:
            raise ValueError('Expected one answer')
        answer = answers[0]
        expected = {'noul': 'predicate', 'choice': 'choice', 'score': 'score'}[task.question['type']]
        if answer.get('name') != 'decision' or answer.get('type') != expected:
            raise ValueError('Refusal or wrong answer type/name')
        valid = lambda p: not isinstance(p, bool) and isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
        if expected == 'predicate':
            p = answer.get('probability')
            if not valid(p): raise ValueError('Invalid predicate probability')
            probabilities = {'yes': p, 'no': 1 - p}
        else:
            probabilities = {}
            for row in answer.get('probabilities') or []:
                k, p = str(row['value']), row['probability']
                if k in probabilities or not valid(p): raise ValueError('Duplicate or invalid probability')
                probabilities[k] = p
        if set(probabilities) != set(task.labels) or abs(sum(probabilities.values()) - 1) > .02:
            raise ValueError('Incomplete/non-normalized native distribution')
        return probabilities

    def run(self, task):
        result = BenchmarkResult(model=MODEL, probs_source='native_decisions_api', missing_policy='error')
        started = time.perf_counter()
        try:
            result.request_body = self.build(task)
            result.raw = self.client.post(result.request_body)
            result.status = 200
            result.usage = result.raw.get('usage') or {}
            usage = result.usage
            if usage.get('output_tokens') != 0 or (usage.get('output_tokens_details') or {}).get('reasoning_tokens') != 0:
                raise ValueError('Zero output/reasoning tokens not verified')
            result.probs = self.decode(result.raw, task)
            result.ok = True
        except urllib.error.HTTPError as error:
            result.status, result.error_kind, result.error = error.code, 'transport', 'HTTPError'
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.error_kind, result.error = 'distribution' if result.status == 200 else 'transport', type(error).__name__
        finally:
            result.latency_s = time.perf_counter() - started
        return result
