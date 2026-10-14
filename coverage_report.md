# Luna adapter validation

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

## Native Decisions API adapter

15 October 2026. This is a separate paid-API adapter, not the subscription implementation above.

- Full suite after adding Decisions: **120 tests PASS; 4 pre-existing opt-in tests skipped**.
- New Decisions code: **156/157 statements (99.36%)**, **37/40 branches (92.50%)**; combined coverage **97.97%**.
- `benchmarks/jevbench_luna_decisions.py`: **100%** combined coverage. `dep_luna_decisions.py`: **97%**.
- VCR replay verifies a real 35-option native distribution. Tests cover predicate/choice/score mappings, gold isolation, native-distribution validation, personal-key selection, pricing/cap behavior, and access/infrastructure stop gates. Model refusals remain wrong answers rather than silently dropping later tasks.
- Live end-to-end run and audit are documented in `docs/benchmarks/luna6_decisions_public_20261015.md`.

```sh
python -m coverage run --branch --source=dep_luna_decisions,benchmarks.jevbench_luna_decisions -m unittest discover -s tests
python -m coverage report -m
python -m coverage html
```
