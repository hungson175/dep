# Luna subscription adapter validation

15 October 2026. Coverage applies to the **new Luna adapter and benchmark runner**, not the entire repository.

- Full suite: **114 tests, PASS; 4 pre-existing opt-in tests skipped**.
- New code: **142/145 statements (97.93%)**, **32/36 branches (88.89%)**; combined coverage **96.13%**.
- Per-file combined coverage: `dep_luna.py` **96%**; `benchmarks/jevbench_luna.py` **97%**.
- VCR replay covers a real public-case subscription response with credential headers excluded. Unit tests cover all three task types, gold isolation, model/reasoning checks, malformed answers, access failures and stop behavior.
- Disk HTML report: `htmlcov/index.html` (generated, gitignored).
- Live 231-case end-to-end evidence is retained separately under `benchmark_runs/`; see the Luna benchmark report for its completed-run outcome.

Exact test commands:

```sh
python -m coverage run --branch --source=dep_luna,benchmarks.jevbench_luna -m unittest discover -s tests
python -m coverage report -m
python -m coverage html
```

No UI, deployed service, dependency installation, model fallback, or paid API route was changed. The new subscription client is experimental, owner-local, and returns labels only; no candidate probability distribution is fabricated.
