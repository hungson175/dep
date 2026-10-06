"""Small, dependency-free public API over the notebook's one-token engine."""
from __future__ import annotations

import os

from deepseek_flash import DeepSeekFlashClient, build_request, confidence


class DeepSeek:
    """Use an explicit key or exported DEEPSEEK_API_KEY; never reads .env files.

    Missing options get 0% by default. Use missing_policy="error" to reject a
    censored distribution instead. Calls are serial and never automatically retried.
    """

    def __init__(self, api_key: str | None = None, *, missing_policy: str = "zero", timeout: float = 120):
        key = os.environ.get("DEEPSEEK_API_KEY", "") if api_key is None else api_key
        self._client = DeepSeekFlashClient(key, timeout_s=timeout, missing_policy=missing_policy)

    def __repr__(self):
        return f"DeepSeek(model={self._client.model!r}, missing_policy={self._client.missing_policy!r})"

    def predict(self, state, questions: dict) -> dict:
        """Answer named noul/choice/score questions with one call per question.

        Validate every question before sending anything. A later API failure raises;
        calls already completed may still be billed (there is no transaction/retry).
        """
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty dictionary")
        for name, q in questions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("question names must be nonempty strings")
            build_request(state, q, self._client.model)
        answers, usages = {}, []
        for name, q in questions.items():
            decision = self._client.evaluate(state, q)
            p, qtype = decision.probabilities, q["type"]
            answer = {"type": qtype, "probabilities": p, "confidence": confidence(p)}
            if qtype == "noul":
                answer["noul"] = p["true"]
            elif qtype == "choice":
                answer["choice"] = max(p, key=p.get)
            else:
                answer["score"] = sum(int(k) * v for k, v in p.items())
                answer["legend"] = {str(i + 1): desc for i, desc in enumerate(q["criteria"])}
            answer["diagnostics"] = {
                "missing_options": decision.missing_options,
                "missing_policy": self._client.missing_policy,
                "candidate_mass": decision.candidate_mass,
            }
            answers[name] = answer
            usages.append(decision.usage)
        usage = {}
        for target, source in [("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                               ("total_tokens", "total_tokens")]:
            values = [u.get(source) for u in usages]
            known = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values)
            usage[target] = sum(values) if known else None
        return {"answers": answers, "model": self._client.model, "usage": usage}

    def noul(self, state, instructions: str, criteria: dict | None = None) -> dict:
        """Return P(true), the complete true/false map, and margin confidence."""
        return self.predict(state, {"decision": {"type": "noul", "instructions": instructions,
                                                 "criteria": criteria}})["answers"]["decision"]

    def choice(self, state, instructions: str, options: dict | list) -> dict:
        """Pick a supplied option and return probabilities over every option."""
        return self.predict(state, {"decision": {"type": "choice", "instructions": instructions,
                                                 "criteria": options}})["answers"]["decision"]

    def score(self, state, instructions: str, levels: list) -> dict:
        """Return the expected 1-based level and its full probability map."""
        return self.predict(state, {"decision": {"type": "score", "instructions": instructions,
                                                 "criteria": levels}})["answers"]["decision"]
