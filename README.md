# studio-chatbot

FastAPI + LangChain chatbot for a fitness studio. FAQ knowledge in S3
is chunked and embedded into pgvector for RAG; an incoming message is routed to one of four
named agents — **membership_registration**, **support**, **general** — each grounded in the
FAQ via a shared retrieval tool, plus **summarize**, a cheap-tier agent with no FAQ tool for
tl;dr-style requests or recapping a session once it's marked **resolved**. Each `/chat` call
returns a `session_id` (server-side, in-memory history) that the client passes back on
subsequent calls so summarize has a transcript to work with.

## Architecture

```
POST /chat --> Router (keyword short-circuit, then LLM classifier)
                 |
                 +--> MEMBERSHIP_REGISTRATION agent --> search_studio_faq tool --> pgvector
                 +--> SUPPORT agent                 --> search_studio_faq tool --> pgvector
                 +--> GENERAL agent                  --> search_studio_faq tool --> pgvector
                 +--> SUMMARIZE agent (gpt-5-nano, no tools)
```

- `app/faq_loader.py` — pulls `faq.md` (S3 in prod — the single source of truth; `data/faq.md`
  in local, gitignored, not checked in, so place your own copy there before running locally),
  splits it by markdown header (one chunk per FAQ section), and ingests into pgvector. Idempotent
  on a content hash — re-running with unchanged FAQ text is a no-op; changed text deletes and
  re-ingests.
- `app/router.py` — cheap keyword short-circuit (join/register/membership/etc → membership
  registration) then a temperature-0 LLM classifier for anything ambiguous.
- `app/agents/` — one system prompt per named agent, all sharing the same FAQ retrieval tool
  and tool-calling loop (`app/assistant.py`).
- `app/session_store.py` — in-memory, per-process history keyed by `session_id`; single pod
  today, so history doesn't survive a restart or span replicas (fine for summarizing a live
  session; would need a shared store like Redis if that mattered).

## Embedding models: local vs. prod

Both profiles use a same-family embedding model, but **not the same vectors or table** — a
pgvector column is fixed-dimension, so local (`sentence-transformers/all-MiniLM-L6-v2`,
384-dim, in-process, no API key needed) and prod (`text-embedding-3-small`, 1536-dim, OpenAI)
each get their own store, re-embedded from the same FAQ source. There is no cross-env vector
sharing.

## Providers

| Component            | local                          | prod                                          |
|-----------------------|---------------------------------|------------------------------------------------|
| Chat / agents / router | OpenAI (`gpt-4o-mini`)         | OpenAI (`gpt-4o-mini`, override via `chatModelName`) |
| Summarize agent         | OpenAI (`gpt-5-nano`)          | OpenAI (`gpt-5-nano`, override via `summarizeModelName`) |
| Embeddings              | HuggingFace (MiniLM, local)    | OpenAI (`text-embedding-3-small`)              |
| Vector store            | pgvector (helm, on kind)        | pgvector (helm, on k8s)                        |

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
- **No tracing.** A single `/chat` call can involve router classification, one or more
  tool-calling round-trips, and a vector search — right now there's no way to see that as
  one trace, only inferred from the single log line's timing. OpenTelemetry
  auto-instrumentation for FastAPI + LangChain would give this without much custom code.
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
  the FAQ RAG loop.
- `evals/test_router_evals.py` — openai/evals-style suite: routing "match" cases (message →
  expected route) and reply "includes" cases (reply must mention phrases actually present in
  the FAQ, i.e. groundedness), structured the way an `openai/evals` YAML eval is shaped
  (input/ideal/grading) without depending on the `openai/evals` package itself.

Both need a live DB + `OPENAI_API_KEY` and are skipped unless explicitly enabled:

```bash
RUN_RAGAS_EVALS=1 RUN_ROUTER_EVALS=1 OPENAI_API_KEY=... pytest evals -q
```

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
  2. Repoint the app at it (`config.dbHost` in `helm/values.yaml`, `helm upgrade
     studio-chatbot ...`) and let the rolling restart happen — each app pod, on startup,
     calls `load_faq_knowledge()` and re-ingests the FAQ from S3/`data/faq.md` fresh into
     `pgvector-v2`. No data ever needs to move between the two Postgres pods.
  3. In prod (`replicaCount: 2`), a standard rolling update keeps one pod serving the old
     DB while the other cuts over — genuinely zero-downtime, briefly split-brained between
     old/new DB, which is harmless since both sides are just re-derivable embeddings. Local
     (`replicaCount: 1`) still has a small gap during its own restart.
  4. Tear down the old `pgvector` release once the new one's confirmed healthy.

  This shortcut depends on the pgvector Postgres staying a pure, re-derivable cache with
  nothing non-reconstructible in it. That's a deliberate boundary, not an accident: session
  state (and later, membership/registration data) is meant to live in a separate
  transactional store behind its own API, not in this Postgres — see "Session state in a
  separate transactional store" in Next Steps below. As long as that boundary holds, this
  blue-green upgrade path stays valid indefinitely.

## Known divergences / simplifications

- "Membership registration" and "support" are chat *advisory* agents grounded in the FAQ, not a
  transactional booking system — matches the FAQ's own guidance ("register via the member
  portal, mobile app, or front desk"); this bot explains policy and process, it doesn't call a
  booking API (there isn't one in scope here).
- Evals are offline/CI-gated only, not runtime guardrails — see "Runtime safety" / "Evals" above.

## Next steps

- **Session state in a separate transactional store, not in-memory.** `app/session_store.py`
  is per-process and lost on pod restart/across replicas — fine for a single pod, not for
  `values-prod.yaml`'s `replicaCount: 2` without session affinity. Deliberately **not** the
  pgvector Postgres — that store stays a pure, re-derivable FAQ cache (see "Evolvability"
  above), which is what makes its zero-downtime blue-green upgrade path possible. Session
  (and, later, membership/registration) state should live behind its own API-backed
  transactional store, so this app calls it over an API rather than holding a direct
  connection to someone else's database -- keeps the two lifecycles (vector cache vs. real
  state) independently upgradable/ownable.
- **Real booking/registration API calls.** Per "Known divergences" above, membership/support
  are advisory-only today. Wiring `MEMBERSHIP_REGISTRATION`'s agent to an actual booking API
  (as a tool, same shape as `search_studio_faq`) would move it from "explains the process" to
  "completes the transaction" -- see the OTP + Keycloak token exchange bullet below for the
  identity piece this needs.
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
  org's existing Keycloak (see `~/mystudio/authapi`) JWKS endpoint, required on `/internal/*`
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
- **On-demand FAQ re-ingestion + pgvector index maintenance (Maintainability).** Today, picking
  up an edited FAQ requires a full pod restart (`lifespan()` re-runs `load_faq_knowledge()` on
  startup) — there's no way to trigger it on demand. Add an internal endpoint (e.g.
  `POST /internal/faq/reindex`, alongside `/internal/faq/search`, same auth boundary once
  Keycloak JWT auth lands above) that calls `load_faq_knowledge()` directly. Also worth
  scripting periodic `VACUUM ANALYZE` on `langchain_pg_embedding` after a re-ingest (the
  delete-then-add pattern in `app/faq_loader.py` churns rows), and checking whether the
  `langchain-postgres` version in use creates an IVFFlat vs HNSW index for this collection --
  IVFFlat degrades more under churn and would benefit from an occasional `REINDEX` more than
  HNSW would. Low priority at this FAQ's current size (single small document), worth revisiting
  if the FAQ corpus grows enough for chunk count or update frequency to matter.
