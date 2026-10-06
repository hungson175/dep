# dep.hungson175.com — model selector deployment

## Approved scope, 2026-10-06

Boss requested live publication, a visitor model selector with **Bonsai default**,
and the server's DeepSeek key with a **hard $20 lifetime site cap**. No official
benchmark submission or paid fast-lane purchase is authorized by this deployment.

## Runtime

- Existing user unit: `dep-playground.service`.
- Checkout/cwd: `/home/hungson175/dev/dep-playground`.
- HTTP: `127.0.0.1:17096`; existing Cloudflare ingress maps
  `dep.hungson175.com` to this listener. No tunnel/DNS changes are needed.
- Existing Bonsai model server: `127.0.0.1:17095`; do not restart it for this release.
- `POST /api/decide` accepts `model: "bonsai"` or `model: "deepseek-flash"`.
  Omitted model remains Bonsai, preserving existing callers.
- DeepSeek uses `dep_deepseek.DeepSeek`, complete zero-filled probability maps,
  a 60-second provider timeout, no retry, and the existing admission/slot/rate gates.
- `POST /api/stress` remains Bonsai-only, including direct API callers.
- No prompts, headers, keys, or user IDs enter the site budget ledger.

## Secret and budget setup

Only the explicitly approved `DEEPSEEK_API_KEY` is provisioned from the primary
credentials store in a child process without printing it. Its private environment
file is `~/.config/dep-playground/deepseek.env` (0600, parent 0700). Never commit
this file or load the entire shared credentials store into the web service.

The unit drop-in `~/.config/systemd/user/dep-playground.service.d/deepseek.conf`
references that environment file and sets `DEP_DEEPSEEK_BUDGET_USD=20` and
`DEP_DEEPSEEK_LEDGER` to `runtime/deepseek_site_ledger.jsonl` in the checkout.
No secret values belong in this document or the unit's inline environment.

The ledger is process-safe through file locking and durable fsync. It reserves
the full 1M-input/one-output-token peak-cost upper bound before each paid call.
Known valid usage settles the reservation; failures, unknown usage, or lost
processes retain it. Restarting does not reset the lifetime cap. Exhaustion or
corrupt evidence fails closed for DeepSeek; Bonsai remains usable. This is a
separate $20 site allowance from the earlier benchmark experiment ledger.

## Release gates

1. Run mocked unit/integration tests; never let tests auto-load local `.env`.
2. Build the minimal library archives and validate their contents using
   `python3 -m scripts.publish_library --output NEW_STAGING_DIRECTORY`.
3. Stage the UI and generated library page/packages outside live `static/`.
   Playwright must verify Bonsai default, explicit DeepSeek selection, disabled
   paid stress, correct request model, mobile layout, and wheel download/install.
4. Commit the tested source/candidate before activation. Retain an exact rollback
   copy of the previous root page and backend while deployment is under review.
5. Copy only `index.html`, `library.html`, and the validated `packages/` release
   into `static/`. Do not expose the repository, secret files, cache, or run logs.
6. Reload the user systemd manager and restart **only** `dep-playground.service`.
7. Verify public health/stats, then one explicit real Bonsai/DeepSeek browser
   journey and public wheel download/install. Do not silently retry a failed
   paid journey; inspect its physical response and ledger first.
8. Remove temporary staging/rollback copies after successful verification.

Library landing page: `/static/library.html`. Release files, checksums and a
machine-readable manifest: `/static/packages/`. These generated artifacts are
ignored by Git; their source template and validating build tool are tracked.

## Rollback

If activation fails, restore only the saved backend/root page for this service,
restart only `dep-playground.service`, and verify health. Retain the paid ledger
and its reservations; never reset budget evidence to make a retry possible.
Do not revert concurrent benchmark-agent commits or restart Bonsai/Cloudflare.

## Bottom summary

- Default: Bonsai, including existing API clients without a model field.
- Optional model: DeepSeek Flash; server-side key, no key field in browser.
- Spending: hard $20 lifetime site cap; unknown billing remains reserved.
- Safety: paid stress blocked; no automatic retries or budget reset.
- Publication: additive library guide/downloads plus live model selector.
- Need from Boss: approval for any future budget increase or official submission.
