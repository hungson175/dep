"""Staging-only providers for browser tests; no credentials or inference."""
import os
from pathlib import Path
from unittest.mock import patch

_exists = os.path.exists
with patch('os.path.exists', side_effect=lambda p: False if Path(p).name == '.env' else _exists(p)), \
        patch.dict(os.environ, {'DEP_API_KEY': 'TEST_ONLY'}):
    import app as module

module.HERE = str(Path(os.environ['DEP_UI_STAGE']).resolve())
from fastapi.staticfiles import StaticFiles
for route in module.app.routes:
    if route.path == '/static': route.app = StaticFiles(directory=f'{module.HERE}/static')
module.STRESS_HASH = 'TEST_ONLY'


def distribution(options):
    names = list(options)
    return {name: 1.0 if i == 0 else 0.0 for i, name in enumerate(names)}


module.minijev.choice = lambda state, instruction, options: (distribution(options), 1.)
module.minijev.model_seconds_reset = lambda: None
module.minijev.model_seconds = lambda: .001


class StagingBudget:
    def snapshot(self): return {'cap_usd': 20, 'charged_usd': 0, 'remaining_usd': 20}


class StagingDeepSeek:
    available = True
    budget = StagingBudget()

    def run(self, state, q):
        p = distribution(q['criteria'])
        return ({'type': 'choice', 'choice': max(p, key=p.get), 'probabilities': p,
                 'confidence': 1., 'diagnostics': {'missing_options': []}},
                {'input_tokens': 10, 'output_tokens': 1, 'total_tokens': 11}, .0000042)


module._deepseek = StagingDeepSeek()
app = module.app
