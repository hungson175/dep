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

## DeepSeek letters52 mode

15 October 2026. Explicit opt-in `marker_scheme="letters52"`; legacy prompts and
35-marker mode remain unchanged. No deployed service or GitHub release replaced.

- Full suite: **128 tests PASS; 3 pre-existing opt-in tests skipped**. Installed-wheel E2E enabled.
- Coverage scope: `deepseek_flash`, `dep_deepseek.client`; **299/302 statements**, **92/96 branches**, combined **98.24%**. Public client100%, engine98%. This is a different scope from the Luna coverage above, not a coverage drop comparison.
- Test-first evidence: five new regression tests failed before implementation; raw-logprob-ranking regression also failed before addition. Eight new tests now pass, covering case separation, no whitespace merge, 49-way transport, stable prefix/question-last layout, censoring/strict mode, raw ranking, cache usage and unknown-field handling.
- Integration: sanitized VCR replay of two real synthetic49-option calls. Live public E2E selected category0 and category26 correctly; second call reported1,664cache-hit/153miss tokens. Only7/9bare markers were reported, so both top10 rankings are explicitly incomplete.
- Offline official V4 tokenizer check:52bare markers eachone distinct token, exact roundtrip; hashes and token IDs recorded in `tests/fixtures/deepseek_v4_letters52_markers.json`. This is offline-tokenizer evidence, not a serverrevision attestation.
- Installed-wheel E2E exercised both legacy and letters52 APIs outside the checkout, no live call. Disk HTML coverage: `htmlcov/index.html`.

```sh
uv build --offline --out-dir dist
DEP_TEST_WHEEL="$PWD/dist/dep_deepseek-0.1.0-py3-none-any.whl" python -m coverage run --branch --source=deepseek_flash,dep_deepseek.client -m unittest discover -s tests
python -m coverage report -m
python -m coverage html
```

Remaining uncovered engine paths are pre-existing list-label validation,
malformed-row handling and parse-failure handling. No runtime dependencies added;
`tokenizers` is optional only for the standalone offline verification tool.
