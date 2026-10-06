"""Paid playground backend with a durable, process-safe $20 lifetime cap.

No credential-file reads, provider retries, or automatic budget resets. Failed
calls and unknown usage keep their maximum-cost reservation. Ledger stores no
prompts, keys, or user identifiers. Separate from the benchmark experiment ledger.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import threading
import uuid

from dep_deepseek import DeepSeek
from deepseek_flash import build_request

# Peak tariff upper bounds checked 2026-10-06; no cache discount assumed.
MAX_INPUT = 1_000_000
INPUT_PER_MILLION, OUTPUT_PER_MILLION = .3, 1.2
RESERVE_USD = (MAX_INPUT * INPUT_PER_MILLION + OUTPUT_PER_MILLION) / 1e6


class BudgetUnavailable(RuntimeError):
    """Budget exhausted or its durable evidence is unusable; never call provider."""


def _number(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


class SiteBudget:
    def __init__(self, path, cap=20):
        if not _number(cap, positive=True) or cap > 20:
            raise ValueError('Site budget must be positive and at most $20')
        self.path, self.cap = Path(path), float(cap)

    def _operation(self, event=None):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a+', encoding='utf-8') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            held, settled = {}, set()
            try:
                for line in stream:
                    row = json.loads(line)
                    rid, cost, kind = row['id'], row['usd'], row['event']
                    if not isinstance(rid, str) or not _number(cost): raise ValueError()
                    if kind == 'reserve' and rid not in held and cost > 0:
                        held[rid] = cost
                    elif kind == 'settle' and rid in held and rid not in settled and cost <= held[rid]:
                        held[rid] = cost; settled.add(rid)
                    else: raise ValueError()
            except (ValueError, KeyError, TypeError):
                raise BudgetUnavailable('DeepSeek budget evidence is invalid; use Bonsai') from None
            charged = math.fsum(held.values())
            if charged > self.cap:
                raise BudgetUnavailable('DeepSeek budget is exhausted; use Bonsai')
            if event is not None:
                rid, cost = event['id'], event['usd']
                if event['event'] == 'reserve':
                    if charged + cost > self.cap:
                        raise BudgetUnavailable('DeepSeek $20 site budget is exhausted or reserved; use Bonsai')
                elif rid not in held or rid in settled or cost > held[rid]:
                    raise BudgetUnavailable('Invalid budget settlement')
                stream.seek(0, os.SEEK_END)
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush(); os.fsync(stream.fileno())
            return {'cap_usd': self.cap, 'charged_usd': charged,
                    'remaining_usd': max(0, self.cap - charged)}

    def reserve(self, cost=RESERVE_USD):
        if not _number(cost, positive=True): raise ValueError('Invalid reservation')
        rid = uuid.uuid4().hex
        self._operation({'event': 'reserve', 'id': rid, 'usd': cost})
        return rid

    def settle(self, rid, cost):
        if not _number(cost): raise ValueError('Invalid settlement')
        self._operation({'event': 'settle', 'id': rid, 'usd': cost})

    def snapshot(self): return self._operation()


class DeepSeekSite:
    def __init__(self, budget, *, key=None):
        self.budget = budget
        self._key = os.environ.get('DEEPSEEK_API_KEY', '') if key is None else key
        self._client, self._lock = None, threading.Lock()

    @property
    def available(self): return bool(self._key and self._key.strip())

    def run(self, state, question):
        build_request(state, question)  # Validate before reserving or billing.
        if not self.available:
            raise RuntimeError('DeepSeek is not configured; use Bonsai')
        with self._lock:
            if self._client is None:
                self._client = DeepSeek(api_key=self._key, timeout=60)
        rid = self.budget.reserve()
        # Any failure preserves the full reservation: billing may be unknown.
        result = self._client.predict(state, {'decision': question})
        usage = result['usage']
        i, o = usage.get('input_tokens'), usage.get('output_tokens')
        valid = lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 0
        cost = None
        if valid(i) and valid(o) and i <= MAX_INPUT and o <= 1:
            cost = (i * INPUT_PER_MILLION + o * OUTPUT_PER_MILLION) / 1e6
            self.budget.settle(rid, cost)
        return result['answers']['decision'], usage, cost
