"""Run the pinned, public authored JevBench cases with the notebook adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request

from deepseek_flash import BenchmarkResult, DeepSeekFlashClient, JevBenchAdapter

PIN = "bb05a335bc809e61b20c0f745d25499a82b326fc"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".benchmark_cache" / "jevbench" / PIN
SCOPE = "public_authored_legacy_not_official_v1.6"
DATASETS = ["datasets/public/easy.jsonl", "datasets/public/original.jsonl", "datasets/public/hard.jsonl"]
FILES = ["LICENSE", "jevbench/__init__.py", "jevbench/tasks.py", "jevbench/scoring.py",
         "jevbench/metrics.py", "jevbench/summarize.py", "jevbench/budget.py"] + DATASETS
# Official peak rates, checked 2026-10-06; conservative even during off-peak.
# https://api-docs.deepseek.com/quick_start/pricing/
PRICE_INPUT, PRICE_HIT, PRICE_OUTPUT = .3, .006, 1.2
MAX_INPUT_TOKENS = 1_000_000


def blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch_source(source=SOURCE):
    """Explicit download only. Verify every file against the pinned Git tree."""
    source = Path(source)
    if source.exists():
        activate_source(source)
        return
    base = "https://api.github.com/repos/fstandhartinger/jevbench"
    with urllib.request.urlopen(base + f"/git/trees/{PIN}?recursive=1", timeout=60) as r:
        tree = json.load(r)
    hashes = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}
    contents = {}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{PIN}/{name}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if blob_sha(data) != hashes[name]:
            raise ValueError(f"upstream blob hash mismatch: {name}")
        contents[name] = data
    source.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_json(source / "source_manifest.json", {
        "pin": PIN, "repository": "https://github.com/fstandhartinger/jevbench",
        "blobs": {name: hashes[name] for name in FILES},
    })


def activate_source(source=SOURCE):
    source = Path(source).resolve()
    manifest = json.loads((source / "source_manifest.json").read_text())
    if manifest["pin"] != PIN or set(manifest["blobs"]) != set(FILES):
        raise ValueError("upstream source manifest mismatch")
    for name, sha in manifest["blobs"].items():
        if blob_sha((source / name).read_bytes()) != sha:
            raise ValueError(f"upstream source changed: {name}")
    if "jevbench" in sys.modules:
        loaded = Path(sys.modules["jevbench"].__file__).resolve()
        if not loaded.is_relative_to(source):
            raise ValueError("a different jevbench source is already imported")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def load_public_tasks(source=SOURCE):
    activate_source(source)
    from jevbench.tasks import load_jsonl
    tasks = [t for name in DATASETS for t in load_jsonl(str(Path(source) / name))]
    if len({t.id for t in tasks}) != len(tasks) or any(t.split != "public" for t in tasks):
        raise ValueError("duplicate or non-public upstream tasks")
    return tasks


def validate_cap(cap):
    if not math.isfinite(cap) or not 0 < cap <= 20:
        raise ValueError("cap must be positive and at most Boss's approved $20")
    return cap


def budget_reserve():
    return (MAX_INPUT_TOKENS * PRICE_INPUT + PRICE_OUTPUT) / 1e6


def measured_cost(result):
    """Bill successful HTTP replies even when candidate parsing failed."""
    usage = result.usage
    i, o = usage.get("input_tokens"), usage.get("output_tokens")
    valid = lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 0
    if not valid(i) or not valid(o):
        return None
    hit, miss = usage.get("prompt_cache_hit_tokens"), usage.get("prompt_cache_miss_tokens")
    if valid(hit) and valid(miss) and hit + miss == i:
        return (hit * PRICE_HIT + miss * PRICE_INPUT + o * PRICE_OUTPUT) / 1e6
    return (i * PRICE_INPUT + o * PRICE_OUTPUT) / 1e6


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run_public(tasks, adapter, ledger, output):
    from jevbench.budget import BudgetExceeded
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize
    from jevbench.tasks import dataset_hash

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    raw_dir = output / "raw"
    raw_dir.mkdir(mode=0o700)
    write_json(output / "manifest.json", {
        "scope": SCOPE, "upstream_pin": PIN, "dataset_hash": dataset_hash(tasks),
        "planned": len(tasks), "recipe": "learn/deepseek-flash.ipynb",
        "implementation_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["deepseek_flash.py", "benchmarks/jevbench_flash.py", "learn/deepseek-flash.ipynb"]
        },
        "settings": {"max_tokens": 1, "temperature": 1.0, "thinking": {"type": "disabled"},
                     "logprobs": True, "top_logprobs": 20, "logit_bias": "omitted"},
        "tariff_upper_bound_usd_per_million": {"input_miss": PRICE_INPUT,
                                              "input_hit": PRICE_HIT, "output": PRICE_OUTPUT},
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        "price_checked": "2026-10-06", "no_retries": True,
    })
    rows, transport_errors, stop_reason = [], 0, None
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for t in tasks:
            try:
                rid = ledger.reserve(budget_reserve(), {"task_id": t.id, "output": str(output)})
            except BudgetExceeded:
                stop_reason = "budget"
                break
            try:
                result = adapter.run(t)
            except Exception as e:
                result = BenchmarkResult(error_kind="transport", error=type(e).__name__)
            # Preserve evidence before settlement or scoring; never include headers/keys.
            raw = {"request": result.request_body, "response": result.raw, "status": result.status}
            write_json(raw_dir / (hashlib.sha256(t.id.encode()).hexdigest() + ".json"), raw)
            scored = score_task(result.probs or {}, t)
            cost = measured_cost(result)
            record = {
                "task_id": t.id, "family": t.family, "split": t.split,
                "question_type": t.question["type"], "ok": result.ok,
                **scored, "probs_source": result.probs_source, "model": result.model,
                "error": result.error, "error_kind": result.error_kind,
                "status_code": result.status, "latency_s": result.latency_s,
                "usage": result.usage, "candidate_mass": result.candidate_mass,
                "cost_usd": cost, "cost_basis": "peak_tariff_upper_bound" if cost is not None else "unknown",
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(record)
            ledger.settle(rid, cost if cost is not None else budget_reserve())
            print(f"{len(rows)}/{len(tasks)} {t.id}: {'valid' if scored['valid'] else 'failed'}", flush=True)
            transport_errors = transport_errors + 1 if result.error_kind == "transport" else 0
            if result.status in (401, 403, 429) or transport_errors >= 3:
                stop_reason = "access_or_transport"
                break
    summary = summarize(tasks, rows, ledger_charged=ledger.charged)
    summary.update(scope=SCOPE, upstream_pin=PIN, stop_reason=stop_reason,
                   official_score=None, official_rank=None,
                   calibration_note="Conditional logprobs, not fitted/validated calibration.",
                   price_note="Peak-tariff upper bound, not a measured provider bill.")
    summary["distribution_failures"] = sum(r["error_kind"] == "distribution" for r in rows)
    summary["missing_candidate_failures"] = sum(
        r["error_kind"] == "distribution" and "missing from top-20" in (r["error"] or "") for r in rows
    )
    summary["per_type"] = {
        kind: summarize([t for t in tasks if t.question["type"] == kind],
                        [r for r in rows if r["question_type"] == kind])
        for kind in sorted({t.question["type"] for t in tasks})
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["fetch", "dry-run", "run"])
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--cap-usd", type=float, default=20)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    if args.action == "fetch":
        fetch_source(args.source)
        print(f"Pinned public source ready: {args.source}")
        return 0
    validate_cap(args.cap_usd)
    tasks = load_public_tasks(args.source)
    if args.limit is not None:
        if args.limit < 1:
            ap.error("--limit must be positive")
        tasks = tasks[:args.limit]
    if args.action == "dry-run":
        # Prepare without a real key or any inference, including all canonical mappings.
        adapter = JevBenchAdapter(DeepSeekFlashClient("DRY_RUN_NO_KEY"))
        for t in tasks:
            adapter.prepare(t)
        print(json.dumps({"scope": SCOPE, "planned": len(tasks), "upstream_pin": PIN,
                          "cap_usd": args.cap_usd, "calls_sent": 0}))
        return 0
    client = DeepSeekFlashClient.from_env()
    from jevbench.budget import Ledger
    output = args.output or ROOT / "benchmark_runs" / time.strftime("%Y%m%dT%H%M%S")
    # Shared across runs: a lost process or new output folder cannot reset spend.
    ledger = Ledger(ROOT / "benchmark_runs" / "ledger.jsonl", args.cap_usd)
    summary = run_public(tasks, JevBenchAdapter(client), ledger, output)
    print(json.dumps({"output": str(output), "complete": summary["complete"],
                      "accuracy": summary["accuracy"], "budget_charged_usd": ledger.charged}))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
