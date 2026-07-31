# studio-chatbot

FastAPI + LangChain chatbot for a fitness studio. FAQ knowledge in a Google Doc — plus
gym-equipment photos, captioned and embedded the same way — is chunked and embedded into
pgvector for RAG; an incoming message is routed to one of four named agents —
**membership_registration**, **support**, **general** — each grounded in the FAQ via a
shared retrieval tool, plus **summarize**, a cheap-tier agent with no FAQ tool for
tl;dr-style requests or recapping a session once it's marked **resolved**. Each `/chat` call
returns a `session_id` (server-side, in-memory history) that the client passes back on
subsequent calls so summarize has a transcript to work with. `membership_registration` also
has a real (read-only) membership-status lookup tool against a sibling service, on top of
FAQ-grounded policy advice. System prompts live in the LangSmith Prompt Hub, not hardcoded,
so CSR staff can edit wording without a deploy — gated to prod via an eval-based promotion
check (see "Prompt management" below).

## Status: what's built so far

- **Core chat + RAG.** Four named agents (`membership_registration`, `support`, `general`,
  `summarize`) behind a keyword-then-LLM router; FAQ retrieval grounds three of them via a
  shared LangChain tool over pgvector; `summarize` recaps a session on trigger words like
  "resolved"/"tldr" using server-side session history (`app/session_store.py`).
- **FAQ pipeline, decoupled from the app pod.** The FAQ source is a Google Doc (service
  account auth, `app/faq_loader.py`) — the product team edits it directly, no deploy needed
  to publish a change. Embedding/ingestion runs offline as a Kubernetes `Job`
  (`scripts/ingest_faq.py`, `ingestJob.enabled`), idempotent on a whole-document content hash.
  The app pod itself does no FAQ freshness check at startup — a per-pod Drive
  call doesn't scale with replica count/restarts, so ingestion is purely the
  offline job's concern.
- **Multimodal RAG for equipment photos.** Gym-equipment images are captioned once by a
  vision-capable chat model and the caption text is embedded into the same pgvector store
  as the FAQ (`app/image_loader.py`, `scripts/ingest_equipment_image.py`) — retrieval quality
  is bounded by caption quality, not pixel similarity. Idempotent on image content hash, same
  delete-then-add pattern as the FAQ loader; a distinct `image://...` source so the two
  pipelines' stale-chunk cleanup never collide.
- **FAQ retrieval result caching.** `app/tools/faq.py` memoizes `(query, k)` similarity
  searches in-process (`functools.lru_cache`) — safe for the process lifetime since FAQ
  content only changes via a fresh ingest + redeploy, never mid-process.
- **Prompt management via LangSmith Hub, with an eval-gated promotion path.** Each agent's
  system prompt is pulled from the Hub at a pinned ref (`app/prompts.py`,
  `settings.<route>_prompt_ref`), not hardcoded, so CSR/support staff can edit prompt wording
  without an engineer touching code. Prod pins a specific commit hash rather than tracking
  "latest," and `scripts/promote_prompt.py` is the gate: it runs a candidate prompt edit
  through both a judge-scored eval suite and RAGAS (faithfulness/answer-relevancy/context-
  precision, using a separate stronger judge model than the one generating replies, to avoid
  self-preference bias and false-positive zeroing) before saying whether it's safe to pin.
  Falls back to hardcoded constants in `app/agents/*.py` if the Hub is unreachable.
- **Real (read-only) membership status lookup.** `MEMBERSHIP_REGISTRATION` can call
  `lookup_membership_status` (`app/tools/studio_api.py`) against a sibling membership
  service, by exact email or phone only (never by name) — narrows, but doesn't fully close,
  the lack of caller identity on an unauthenticated `/chat`. Still advisory for anything
  that *changes* a membership — see "Known divergences" below.
- **Runtime safety.** Every `/chat` message is checked against OpenAI's Moderation API before
  reaching any agent or the session store; flagged messages get a 400.
- **Observability.** Every chat turn is logged in one line (session_id, route, message, reply);
  gaps (metrics, tracing, alerting) are named explicitly rather than silently absent — see
  "Observability" below.
- **Evals.** Offline/CI-gated router-accuracy and RAGAS (faithfulness/relevancy/context-precision)
  suites, kept separate from runtime guardrails.
- **Deployment.** Helm charts for both the app and pgvector (StatefulSet), deployed and
  exercised against a local `kind` cluster and structured for prod (`values-prod.yaml`,
  `replicaCount: 2`, IRSA-ready service account, Secrets Manager for DB creds).
- **Evolvability.** A documented, validated zero-downtime pattern for both pgvector version
  upgrades and FAQ content changes: stand up a fresh pgvector release, (re-)ingest into it
  offline, cut the app over via `config.dbHost`, tear down the old release — safe because
  pgvector here is a pure, re-derivable FAQ cache, never primary data (see "Evolvability"
  below).
- **Simplification.** The one-time DB migration mechanism was removed entirely (single
  release, `CREATE EXTENSION vector` lives in pgvector's own init script) rather than kept
  around for a migration path that doesn't exist yet.
- **Explicitly not done yet** — real booking/registration API calls, Keycloak-based auth,
  session state in a real transactional store, and everything else tracked in "Next steps"
  below; these are scoped decisions, not oversights.

## Architecture

See [docs/WORKFLOWS.md](docs/WORKFLOWS.md) for detailed flowcharts of each code path
(startup/FAQ ingestion, `POST /chat`, `GET /internal/faq/search`).

```
POST /chat --> Router (keyword short-circuit, then LLM classifier)
                 |
                 +--> MEMBERSHIP_REGISTRATION agent --> search_studio_faq tool --> pgvector
                 |                                  --> lookup_membership_status tool --> membership service
                 +--> SUPPORT agent                 --> search_studio_faq tool --> pgvector
                 +--> GENERAL agent                  --> search_studio_faq tool --> pgvector
                 +--> SUMMARIZE agent (gpt-5-nano, no tools)

data/faq.md + equipment photos --> pgvector (offline ingest, see below)
```

- `app/faq_loader.py` — pulls `faq.md` (a Google Doc in prod — the single source of truth, so
  the product team can edit it directly, no deploy step; `data/faq.md` in local, gitignored,
  not checked in, so place your own copy there before running locally), splits it by markdown
  header (one chunk per FAQ section), and ingests into pgvector. Idempotent on a content hash
  — re-running with unchanged FAQ text is a no-op; changed text deletes and re-ingests. Runs
  offline (`scripts/ingest_faq.py`, a Kubernetes `Job`), not from the app pod's own startup —
  see `docs/WORKFLOWS.md` workflows 1 and 1b.
- `app/image_loader.py` — captions gym-equipment photos with a vision-capable chat model and
  embeds the caption text into the same pgvector store, same content-hash idempotency and
  delete-then-add pattern as the FAQ loader. Runs offline (`scripts/ingest_equipment_image.py`),
  one image at a time.
- `app/router.py` — cheap keyword short-circuit (join/register/membership/etc → membership
  registration) then a temperature-0 LLM classifier for anything ambiguous.
- `app/prompts.py` — resolves each agent's system prompt from the LangSmith Prompt Hub at a
  pinned ref (`settings.<route>_prompt_ref`), falling back to the hardcoded constants in
  `app/agents/*.py` if the Hub is unreachable.
- `app/agents/` — one system prompt per named agent (pulled via `app/prompts.py`), all
  sharing the FAQ retrieval tool and tool-calling loop (`app/assistant.py`);
  `MEMBERSHIP_REGISTRATION` additionally gets `lookup_membership_status`
  (`app/tools/studio_api.py`).
- `app/tools/faq.py` — the shared FAQ retrieval tool; caches `(query, k)` similarity-search
  results in-process for the life of the pod.
- `app/tools/studio_api.py` — calls the sibling membership service's
  `GET /api/memberships/lookup` (email or phone only, never by name) for real membership
  status, sandboxed so `/chat` can't be used to browse other members' records by name.
- `app/session_store.py` — in-memory, per-process history keyed by `session_id`; single pod
  today, so history doesn't survive a restart or span replicas (fine for summarizing a live
  session; would need a shared store like Redis if that mattered).

## Embedding models: local vs. prod

Both profiles use the same OpenAI embedding model (`text-embedding-3-small`, 1536-dim), but
**not the same vectors or table** — each profile points at its own DB and re-embeds from the
same FAQ source independently. There is no cross-env vector sharing.

## Providers

| Component            | local                          | prod                                          |
|-----------------------|---------------------------------|------------------------------------------------|
| Chat / agents / router | OpenAI (`gpt-4o-mini`)         | OpenAI (`gpt-4o-mini`, override via `chatModelName`) |
| Summarize agent         | OpenAI (`gpt-5-nano`)          | OpenAI (`gpt-5-nano`, override via `summarizeModelName`) |
| Equipment image captioning | OpenAI (`gpt-4o-mini`, vision-capable) | same |
| Eval/promotion RAGAS judge | OpenAI (`gpt-4o`, `scripts/promote_prompt.py` / `evals/test_ragas_faq.py`) | same |
| Embeddings              | OpenAI (`text-embedding-3-small`) | OpenAI (`text-embedding-3-small`)           |
| Vector store            | pgvector (helm, on kind)        | pgvector (helm, on k8s)                        |
| Prompt source            | LangSmith Prompt Hub (`ref="latest"`) | LangSmith Prompt Hub (pinned commit hash per route) |
| Membership status lookup | sibling `membership` service (docker-compose, `localhost:8082`) | sibling `membership` service (`membershipApiBaseUrl`) |

Chat/agents/router call OpenAI directly in both profiles. `OPENAI_API_KEY` is required in prod
for both chat and embeddings, plus AWS credentials for S3 and optionally Secrets Manager.

## Local development

Local pgvector + app both run via the Helm charts under `helm/`, on the shared `infra-local`
kind cluster (see `~/projects/infra/local/README.md` for cluster/ingress-nginx setup — this
app just needs the cluster itself, not ingress-nginx or the shared postgres, since it ships
its own pgvector StatefulSet):

```bash
KCTX=kind-infra-local

kubectl --context $KCTX create namespace studio-chatbot   # if it doesn't exist yet

# pgvector uses the public pgvector/pgvector:pg16 image directly -- only the app
# image needs building + loading (kind can't pull from the local docker daemon)
docker build -t senthil-studio-chatbot:latest .
kind load docker-image senthil-studio-chatbot:latest --name infra-local

helm upgrade --install pgvector helm/pgvector -f helm/pgvector/values.yaml \
  --namespace studio-chatbot --kube-context $KCTX
kubectl --context $KCTX -n studio-chatbot rollout status statefulset/pgvector

helm upgrade --install studio-chatbot helm -f helm/values.yaml \
  --set secrets.openaiApiKey=$OPENAI_API_KEY \
  --namespace studio-chatbot --kube-context $KCTX
kubectl --context $KCTX -n studio-chatbot rollout status deployment/studio-chatbot

kubectl --context $KCTX -n studio-chatbot port-forward svc/studio-chatbot 8089:8080
```

```bash
# teardown
helm uninstall studio-chatbot pgvector --namespace studio-chatbot --kube-context $KCTX
```

Or directly, without a cluster at all (fastest inner loop — needs a reachable Postgres,
e.g. the port-forwarded `pgvector` service above, or any local Postgres with the `vector`
extension installed):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env   # Settings (app/config.py) loads .env automatically
uvicorn app.main:app --reload --port 8080
```

```bash
curl -s -X POST http://localhost:8089/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How do I sign up for the yoga class?"}'
# -> {"agent":"...", "reply":"...", "session_id":"..."} — pass session_id back on later
# calls in the same conversation so the summarize agent has history when you say "resolved"

curl -s 'http://localhost:8089/internal/faq/search?q=refund+policy'
```

## Testing

```bash
pip install -r requirements.txt
pytest tests -q          # mocked, no API key / DB needed
```

## Runtime safety

`app/moderation.py` checks every incoming `/chat` message against OpenAI's Moderation API
before it reaches the router or any agent — a flagged message gets a `400` immediately,
nothing is sent to `chat_model`/`summarize_model`. This is the one runtime guardrail in the
request path; `evals/` below is offline quality measurement, not a safety filter.

## Observability

What exists today is minimal: `app/main.py`'s `chat()` logs one line per turn
(`session_id`, `route`, `message`, `reply`) via Python's standard `logging` module to
stdout — `logging.basicConfig(level=logging.INFO)`, no structured/JSON formatting, no log
aggregation configured, no metrics, no tracing. In kind/k8s that means `kubectl logs`, not a
dashboard.

Gaps, in rough priority order:
- **No metrics.** Nothing tracks request volume, latency, route distribution (how often
  each of the four agents fires), moderation-flag rate, or per-provider (OpenAI) error
  rate/cost. A Prometheus counter/histogram per route + a `/metrics` endpoint (e.g.
  `prometheus-fastapi-instrumentator`) would be the cheapest way to close this.
- ~~**No tracing.**~~ **Done via LangSmith.** Set `config.langchainTracing: true` +
  `secrets.langchainApiKey` (Helm) — since every LLM/tool call already runs through
  LangChain runnables, tracing needs no code changes, just `LANGCHAIN_TRACING_V2`,
  `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` env vars. A single `/chat` call's router
  classification, tool-calling round-trips, and vector search all show up as one linked
  trace in the LangSmith UI. Off by default in both `values.yaml` and `values-prod.yaml` —
  before enabling in prod, decide on trace sampling and whether user message content needs
  redaction before being sent to a third-party SaaS (real user text, not just metadata,
  flows through these traces).
- **No structured logs.** The current `log.info(...)` is a formatted string, not JSON — fine
  for `kubectl logs | grep`, harder to query/alert on once there's an actual log pipeline
  (CloudWatch Logs, etc). Worth switching to structured logging (e.g. `structlog`) before
  volume makes grep-based debugging painful.
- **No alerting.** Nothing pages on elevated error rates, moderation-flag spikes, or OpenAI
  API failures — those currently only surface if someone happens to be reading logs.

None of this is wired into CI/CD yet — it'd sit alongside the SAST/DAST pipeline work in
"Next steps" below, not replace it.

## Evals (RAGAS + openai/evals-style, offline/CI-gated)

`evals/` holds **offline quality measurement**, not a runtime guardrail (see "Runtime safety"
above) — these are CI-gated regression checks against `data/faq.md`.

- `evals/test_ragas_faq.py` — RAGAS faithfulness / answer-relevancy / context-precision over
  the FAQ RAG loop. Judges with a separate, stronger model (`gpt-4o`) than the one generating
  replies (`gpt-4o-mini`) — using the same model for both risks self-preference bias, and
  `gpt-4o-mini` as judge was found to zero out genuinely relevant replies over stylistic
  noise (see "Prompt management" above for the concrete failure case). Needs the
  `.venv-py313` interpreter, not whatever Python runs the rest of the suite — `ragas`'s
  `nest_asyncio` dependency is incompatible with Python 3.14's `asyncio.wait_for`/
  `asyncio.timeout` (see `requirements-evals.txt`):
  ```bash
  python3.13 -m venv .venv-evals && source .venv-evals/bin/activate
  pip install -r requirements.txt -r requirements-evals.txt
  ```
  Latest local run (2026-07-31, `SUPPORT` agent, `gpt-4o` judge, against the current
  `data/faq.md`; 7 real cases + 1 `expect_low_relevancy` control from
  `evals/faq_eval_dataset.py` — small enough that a couple of borderline gradings can move
  the mean a fair bit, so treat this as a directional health check, not a leaderboard
  number):

  | Metric | Score | Threshold |
  |---|---|---|
  | Faithfulness | 0.80 | ≥ 0.80 |
  | Answer relevancy | 0.97 | ≥ 0.80 |
  | Context precision | 0.90 | ≥ 0.80 |
  | Control-case relevancy (expect low) | 0.00 | ≤ 0.30 |

  All four passed. Re-run via
  `RUN_RAGAS_EVALS=1 pytest evals/test_ragas_faq.py -q` (needs a live pgvector with the FAQ
  ingested, and `OPENAI_API_KEY`) — the pytest run only asserts pass/fail, not the numbers
  above; get the raw scores by calling `ragas.evaluate(...)` directly the way
  `test_faq_rag_meets_ragas_thresholds` does, or add a `print(main_scores)` locally.
- `evals/test_llm_judge_evals.py` — LLM-as-judge pass/fail scoring over reply quality,
  separate from RAGAS's metric-based scoring.
- `evals/test_router_evals.py` — openai/evals-style suite: routing "match" cases (message →
  expected route) and reply "includes" cases (reply must mention phrases actually present in
  the FAQ, i.e. groundedness), structured the way an `openai/evals` YAML eval is shaped
  (input/ideal/grading) without depending on the `openai/evals` package itself.

These need a live DB + `OPENAI_API_KEY` and are skipped unless explicitly enabled:

```bash
RUN_RAGAS_EVALS=1 RUN_ROUTER_EVALS=1 OPENAI_API_KEY=... pytest evals -q
```

`scripts/langsmith_baseline.py` and `scripts/langsmith_ragas.py` run the same judge/RAGAS
checks against a LangSmith-hosted dataset (`DATASET_NAME`) rather than pytest fixtures — what
`scripts/promote_prompt.py` (see "Prompt management" above) reuses to gate a Hub prompt edit.

## Kubernetes / Helm (prod)

Two charts, deployed separately (`helm/pgvector` first, the root `helm` chart second — the app
expects `DB_HOST` to resolve to the pgvector chart's release name). For local (kind) use, see
"Local development" above — same charts, `values.yaml` instead of `values-prod.yaml`:

```bash
helm install pgvector helm/pgvector -f helm/pgvector/values-prod.yaml
helm install studio-chatbot helm -f helm/values-prod.yaml \
  --set secrets.dbPassword=... \
  --set secrets.openaiApiKey=... \
  --set config.dbSecretName=/rds/... \
  --set config.awsRegion=us-east-1
```

Notes:
- `pgvector` is a single-replica `StatefulSet` with a `PersistentVolumeClaim` — not HA, matches
  the scale of a single FAQ document. Not the `~/projects/infra` RDS module; this app owns its
  own DB lifecycle in-cluster (see the pgvector-location tradeoff discussed with the user before
  building this).
- In prod, the app pod needs IAM permissions for S3 (`s3:GetObject` on
  `senthil-studio-faq/faq.md`) and optionally Secrets Manager — attach via IRSA
  (`serviceAccountName` + annotated `ServiceAccount`, cluster/account-specific, not templated
  here) rather than static keys. Chat/embeddings auth is just `OPENAI_API_KEY`, no AWS
  permissions needed for those.
- `pgvector` runs the public `pgvector/pgvector:pg16` image directly (extension is prebuilt in),
  so it needs no image build/ECR mirror of its own, local or prod.
- The app chart's `values-prod.yaml` has an `<ECR_REPO_URL>` placeholder and no real secrets —
  fill in at deploy time from wherever this org keeps prod credentials, never check them in.

## Evolvability

### Upgrading pgvector

Two different kinds of upgrade, two different risk profiles:

- **Same Postgres major version, newer `vector` extension** (e.g. a new pgvector release
  within the same `pgvector/pgvector:pg16` image family) — in-place, no data loss. Bump
  `image.tag` in `helm/pgvector/values.yaml` (or `values-prod.yaml`), `helm upgrade`, then
  from inside the pod:
  ```sql
  \dx                              -- shows currently installed extension version
  ALTER EXTENSION vector UPDATE;   -- upgrades to whatever's bundled in the new image
  ```
  This only upgrades to whatever version is compiled into the image already running — it
  doesn't fetch a newer extension from anywhere on its own.

- **Postgres major version bump** (e.g. `pg16` → `pg17`) — not in-place. Postgres data
  directories aren't compatible across major versions; swapping the image tag against the
  existing `PersistentVolumeClaim` just gets a pod stuck in `CrashLoopBackOff` on `initdb`
  version mismatch, not silent data loss, but real downtime until fixed. Normally this needs
  a `pg_dump`/restore maintenance window — but **this app's pgvector store is a derived
  cache of the FAQ document, not primary data** (see `app/faq_loader.py`'s content-hash
  idempotency), so a zero-downtime path exists that a general Postgres upgrade wouldn't have:
  1. Stand up a second pgvector release on the new major version, empty
     (`helm install pgvector-v2 helm/pgvector -f helm/pgvector/values.yaml --set image.tag=pgNN ...`),
     under its own release name/Service so it doesn't collide with the running one.
  2. Run the offline ingest Job against `pgvector-v2` (`helm upgrade studio-chatbot ...
     --set ingestJob.enabled=true --set config.dbHost=pgvector-v2`) to populate it fresh
     from the Google Doc/`data/faq.md` source — ingestion is a separate offline concern
     from the app pod now (see `docs/WORKFLOWS.md` workflow 1b), not something app startup
     does on its own.
  3. Repoint the app at it (`config.dbHost` in `helm/values.yaml`, `helm upgrade
     studio-chatbot ...`) and let the rolling restart happen. No data ever needs to move
     between the two Postgres pods.
  4. In prod (`replicaCount: 2`), a standard rolling update keeps one pod serving the old
     DB while the other cuts over — genuinely zero-downtime, briefly split-brained between
     old/new DB, which is harmless since both sides are just re-derivable embeddings. Local
     (`replicaCount: 1`) still has a small gap during its own restart.
  5. Tear down the old `pgvector` release once the new one's confirmed healthy.

  This shortcut depends on the pgvector Postgres staying a pure, re-derivable cache with
  nothing non-reconstructible in it. That's a deliberate boundary, not an accident: session
  state (and later, membership/registration data) is meant to live in a separate
  transactional store behind its own API, not in this Postgres — see "Session state in a
  separate transactional store" in Next Steps below. As long as that boundary holds, this
  blue-green upgrade path stays valid indefinitely.

## Known divergences / simplifications

- "Membership registration" and "support" are chat *advisory* agents grounded in the FAQ, not a
  transactional booking system — matches the FAQ's own guidance ("register via the member
  portal, mobile app, or front desk"); this bot explains policy and process and can look up
  existing membership status (`lookup_membership_status`), but it doesn't call a booking API to
  create/change/cancel anything (there isn't one wired up yet — see "Next steps").
- Evals are offline/CI-gated only, not runtime guardrails — see "Runtime safety" / "Evals" above.

## Prompt management (LangSmith Hub)

Each named agent's system prompt lives in the LangSmith Prompt Hub, not hardcoded — see
`app/prompts.py`. This lets CSR/support staff edit prompt wording directly in the Hub
(Playground or repo page) without an engineer touching `app/agents/*.py` or shipping a deploy.

- **Seeding.** `python -m scripts.push_prompts` pushes the current hardcoded fallback prompts
  to the Hub the first time (`studio-chatbot-support`, `studio-chatbot-general`,
  `studio-chatbot-membership-registration`).
- **Local dev** resolves each prompt at `ref="latest"` by default, so an edit in the Hub
  Playground is immediately visible on the next chat turn — no promotion step needed.
- **Prod pins a specific commit hash**, not "latest" (`settings.<route>_prompt_ref`, set via
  `values-prod.yaml`'s `config.supportPromptRef` / `generalPromptRef` /
  `membershipPromptRef`), so an in-progress CSR edit never reaches prod traffic on its own.
- **Promotion gate.** `scripts/promote_prompt.py <prompt-name>` pulls a prompt's `:latest`
  commit and runs it through both a judge-scored eval suite (reusing
  `scripts/langsmith_baseline.py`'s dataset/evaluator) and RAGAS
  (faithfulness/answer-relevancy/context-precision, same thresholds as
  `evals/test_ragas_faq.py`), then prints the commit hash to pin if both pass. The RAGAS judge
  intentionally uses a *different, stronger* model (`gpt-4o`) than the one generating replies
  (`gpt-4o-mini`) — sharing a model between generator and judge risks self-preference bias, and
  `gpt-4o-mini` as judge was independently found to zero out perfectly relevant replies just for
  ending with a "contact the front desk" pointer (RAGAS's `answer_relevancy` noncommittal
  detector misfiring). Promotion itself (bumping the pinned ref in `values-prod.yaml`) is a
  manual config change — this script only says whether a pending edit is safe to promote. Needs
  the `.venv-py313` interpreter for the RAGAS half (see "Evals" below for why).

## Next steps

- **Human handover.** No path today from a bot conversation to a live human agent — every
  `/chat` call is routed to one of the four LLM agents, with no escalation route out.
  Real chat platforms (a web widget backed by Intercom/Front/Zendesk-style inboxes, or
  SMS/WhatsApp via Twilio) normalize multiple channels into one queue and support handing a
  conversation from bot to human mid-session; this app has neither the trigger nor the
  hand-off mechanics yet. Two pieces needed:
  1. **An escalation trigger.** Either explicit (user says "talk to a person") or implicit
     (moderation-flagged message, tool loop exhausted without resolving anything, a
     business-rule trip like a refund dispute). Natural fit as a fifth `Route` (e.g.
     `Route.HANDOFF`) in `app/router.py`, alongside the existing keyword-short-circuit /
     LLM-classifier pattern already used for `SUMMARIZE`.
  2. **The hand-off itself.** Once triggered, push the session's transcript (already
     available via `SessionStore.transcript()`, the same mechanism `SUMMARIZE` uses) to
     wherever human agents work, via that platform's API — a human needs the full
     conversation, not just the triggering message. The session then needs to be marked
     human-owned so subsequent `/chat` calls for that `session_id` stop routing to an LLM
     agent and instead passthrough (or queue) until a human closes it out or hands it back.
     This human-owned flag is exactly the kind of state that belongs in the transactional
     session store below, not `SessionStore`'s in-memory dict.
- **Session state in a separate transactional store, not in-memory.** `app/session_store.py`
  is per-process and lost on pod restart/across replicas — fine for a single pod, not for
  `values-prod.yaml`'s `replicaCount: 2` without session affinity. Deliberately **not** the
  pgvector Postgres — that store stays a pure, re-derivable FAQ cache (see "Evolvability"
  above), which is what makes its zero-downtime blue-green upgrade path possible. Session
  (and, later, membership/registration) state should live behind its own API-backed
  transactional store, so this app calls it over an API rather than holding a direct
  connection to someone else's database -- keeps the two lifecycles (vector cache vs. real
  state) independently upgradable/ownable.
- **Real booking/registration API calls (write path).** `MEMBERSHIP_REGISTRATION` can already
  look up existing status read-only (`lookup_membership_status`, see "Known divergences"
  above), but still can't create, change, or cancel anything. Wiring a
  `register_for_membership`-style tool (same shape as `lookup_membership_status`) to an actual
  booking API would move it from "explains the process" to "completes the transaction" -- see
  the OTP + Keycloak token exchange bullet below for the identity piece this needs before a
  write path is safe to expose on an unauthenticated `/chat`.
- **Agentic invocation of this API.** Right now every caller is a human via `/chat`. As other
  internal services or agents start calling studio-chatbot programmatically (e.g. an
  orchestrator agent treating this service as one of its own tools), that likely means:
  - a machine-facing auth path (API key/service token) distinct from the human session flow,
  - a stable, documented request/response contract (this README's `/chat` shape becomes a real
    API contract, not just human-readable docs),
  - and probably a dedicated route/flag rather than overloading the same `/chat` + keyword-router
    path both humans and agents hit — an agent calling in doesn't need the keyword short-circuit
    or classifier, it can specify the target agent directly.
- **Session affinity or a shared store, whichever comes first.** Until the transactional
  session store above lands, `sessionAffinity: ClientIP` on the Service is a cheap stopgap so
  `SUMMARIZE`/"resolved" reliably hits the pod holding that session's history in prod.
- **Multi-turn tool loop history.** `Assistant.chat()` only sees the current message plus its
  own tool-call loop, not prior turns — once sessions live in the transactional store above,
  feeding stored history into the agent's own messages (not just the summarize agent) would
  let membership/support/general hold a real multi-turn conversation instead of treating each
  message independently.
- **SAST/DAST in the pipeline.** Nothing currently scans the code, deps, or built image:
  - SAST: [Bandit](https://github.com/PyCQA/bandit) (`bandit -r app/`) and/or
    [Semgrep](https://semgrep.dev/) (`semgrep --config p/python .`) against `app/`;
    [pip-audit](https://github.com/pypa/pip-audit) against `requirements.txt` for known CVEs in
    dependencies (e.g. `langchain`/`fastapi` versions with disclosed vulns).
  - Image scanning for what lands in ECR: [Trivy](https://github.com/aquasecurity/trivy)
    (`trivy image --exit-code 1 --severity HIGH,CRITICAL <image>`) as a CodeBuild gate before
    push, plus turning on ECR enhanced scanning (Inspector-backed) as an always-on backstop
    after push. [Hadolint](https://github.com/hadolint/hadolint) for Dockerfile best-practice
    lint (root user, unpinned base image, etc).
  - DAST: [OWASP ZAP](https://www.zaproxy.org/) baseline scan against a running deploy (kind
    locally or staging) — FastAPI's auto-generated `/openapi.json` lets ZAP target the two real
    endpoints (`/chat`, `/internal/faq/search`) instead of a blind crawl.
  - Wire the SAST/dependency/image steps into CodeBuild before the `docker build`/push stage
    (fail fast); run ZAP post-deploy against staging, never directly against prod.
- **Multi-provider abstraction.** `app/ai_config.py` constructs `ChatOpenAI`/`OpenAIEmbeddings`
  directly, so swapping or adding a provider (Anthropic, Bedrock, etc — evaluated once already
  for prod chat, reverted for now, since Bedrock's `bedrock-runtime` vs `bedrock-mantle` split
  and per-model API-shape differences added more complexity than it was worth mid-refactor)
  means editing model-construction code, not config. If multi-provider ever becomes a real
  requirement (cost fallback, provider outage failover, per-tenant model choice) rather than a
  one-off swap, worth introducing a thin factory keyed off `settings` (provider name + model
  name + any provider-specific kwargs) so `chat_model()`/`embedding_model()` stay call sites,
  not the place provider-specific branching lives. Not worth building speculatively before
  there's a second provider actually in play.
- **Static type checking.** No `mypy`/`pyright` (or similar) runs anywhere in this repo today,
  so type hints are documentation only, not enforced — see the FastAPI Java/Python discussion
  above. Add a checker (as a CI step, same place SAST would run) and reintroduce parameter/
  return annotations (e.g. `app/tools/faq.py`'s `top_k`, `app/main.py`'s `faq_search`'s `topK`,
  currently left unannotated) once there's a checker actually consuming them.
- **Retry transient model-call failures.** Neither `Assistant.chat()`'s `self._model.invoke(...)`
  nor `Router._classify()`'s `self._router_chat_model.invoke(...)` retry on transient OpenAI API
  errors (rate limits, timeouts, connection resets) — today those just propagate as a 500.
  Wrap the actual `.invoke()` calls with [tenacity](https://github.com/jd/tenacity)
  (`@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`, retrying on the relevant
  `openai`/`httpx` exception types) around each model call site, not around
  `Assistant.chat()`'s tool-calling loop as a whole — that loop already has its own bounded
  iteration count (`MAX_TOOL_ITERATIONS`) for a different reason (walking a multi-step tool-use
  conversation to completion, not retrying a failed call) and shouldn't be conflated with retry.
- **Keycloak-based JWT authentication.** `POST /chat` and `GET /internal/faq/search` are both
  unauthenticated today. Add a FastAPI dependency that validates a bearer JWT against this
  org's existing Keycloak (see `~/projects/keycloak`) JWKS endpoint, required on `/internal/*`
  at minimum, `/chat` too if end users should be identified rather than anonymous.
- **Real membership registration via OTP + Keycloak token exchange.** For
  `MEMBERSHIP_REGISTRATION` to actually complete a transaction (not just explain the process --
  see "Known divergences"), it needs to know *who* is registering. Chat has no login step, so
  full interactive Keycloak auth doesn't fit; the flow instead:
  1. A `send_otp(contact)` tool sends a one-time code to the email or phone the user provides
     (SES/SNS, or Twilio for SMS).
  2. A `verify_otp(contact, code)` tool checks it and marks the session (`SessionStore`, or its
     Postgres-backed successor -- see above) as verified for that contact. This only proves
     "controls this email/phone," not a full identity -- fine for *new* registration, weaker
     for looking up an existing member's account.
  3. On success, studio-chatbot's backend service account performs a **Keycloak Token Exchange**
     (RFC 8693 -- not the admin-console "Impersonation" feature, which is cookie/browser-based
     and not meant for backend calls) using the `impersonation` fine-grained permission, to
     obtain a real access token for that user (looked up/created by contact in Keycloak).
  4. `register_for_membership(...)` sends that token downstream, so the booking API sees a
     normal authenticated member request rather than a chatbot-specific bypass.

  Security note: the service account doing the token exchange can impersonate *any* realm
  user -- scope it tightly (only the `impersonation` permission, only fires after a fresh OTP
  verification, short-lived tokens), and log every exchange (ties into the moderation-style
  logging already on `/chat`) rather than running it as a broad admin credential.
- **FAQ re-ingestion is fully offline, not app startup (Maintainability) — mostly done,
  a few gaps remain.** `app.main`'s `lifespan()` does not touch FAQ freshness at all;
  the actual embedding work lives in `scripts/ingest_faq.py`, run via
  `helm/templates/ingest-job.yaml` (`ingestJob.enabled`, off by default) — see
  `docs/WORKFLOWS.md` workflows 1 and 1b. This keeps embedding compute (and the Drive-read
  dependency) out of the request-serving pod entirely. Still open: (1) the Job is triggered
  manually today — a `CronJob` on a schedule, or a webhook off a Drive change notification,
  would close the loop so a stale-content warning doesn't require a human to notice it in
  logs and run the Job by hand; (2) periodic `VACUUM ANALYZE` on `langchain_pg_embedding`
  after a re-ingest (the delete-then-add pattern in `app/faq_loader.py` churns rows); (3)
  checking whether the `langchain-postgres` version in use creates an IVFFlat vs HNSW index
  for this collection — IVFFlat degrades more under churn and would benefit from an
  occasional `REINDEX` more than HNSW would. Low priority at this FAQ's current size (single
  small document), worth revisiting if the FAQ corpus grows enough for chunk count or update
  frequency to matter.
- **Delete-then-add consistency gap during re-ingestion (low priority).** `load_faq_knowledge()`'s
  `_delete_stale(SOURCE_ID)` commits immediately, then `add_documents()` runs moments later as
  a separate call — while the ingest Job is running, a `/chat` or `/internal/faq/search`
  request against that source can briefly see zero results, not stale-but-present ones. Two
  ways to close it, from cheapest to most robust: (1) reorder to add-new-then-delete-old
  instead of delete-then-add, accepting brief duplicate hits over brief empty ones; (2) reuse
  the same **blue-green pgvector pattern already used for version upgrades** (see
  "Evolvability" above) for FAQ *content* changes too — stand up a fresh pgvector release,
  point the offline ingest Job at it instead of the live one, then cut the app over via
  `config.dbHost` once it's fully populated, so the running pgvector the app is actually
  querying is never touched mid-ingest at all. Given this FAQ's current size, the window is
  likely sub-second either way — not worth building until the corpus or ingestion frequency
  grows enough for it to matter.
- **Goal-oriented agent for membership registration.** Every named agent today is reactive:
  one message in, one tool-loop, one reply out (see `docs/WORKFLOWS.md` workflow 2) — there's
  no notion of a multi-step goal the agent is actively working toward across turns. Real
  registration (the OTP + Keycloak token exchange bullet above) is inherently multi-step —
  verify identity, pick a plan, confirm, pay, complete — which a single reactive prompt/tool
  loop handles poorly (nothing tracks "what step is this session on" or retries a stalled
  step). Once session history is in the transactional store (see above), `MEMBERSHIP_REGISTRATION`
  is the natural first candidate for a goal-oriented redesign: an explicit state machine (or a
  planning loop) that knows the registration goal, tracks progress per `session_id`, and only
  calls `register_for_membership(...)` once every precondition (OTP-verified, plan selected,
  payment confirmed) is actually met -- rather than relying on the model to remember and
  sequence all of that correctly inside one `Assistant.chat()` tool-calling loop.
