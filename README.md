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

curl -s 'http://localhost:8089/faq/search?q=refund+policy'
```

## Testing

```bash
pip install -r requirements.txt
pytest tests -q          # hermetic, no API key / DB needed
```

## Evals (RAGAS + openai/evals-style, offline/CI-gated)

`evals/` holds **offline quality measurement**, not runtime guardrails — these are CI-gated
regression checks against `data/faq.md`, not an input/output safety filter in the request path.

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

## Known divergences / simplifications

- "Membership registration" and "support" are chat *advisory* agents grounded in the FAQ, not a
  transactional booking system — matches the FAQ's own guidance ("register via the member
  portal, mobile app, or front desk"); this bot explains policy and process, it doesn't call a
  booking API (there isn't one in scope here).
- Evals are offline/CI-gated only, not runtime guardrails — see "Evals" above.

## Next steps

- **Session state in Postgres, not in-memory.** `app/session_store.py` is per-process and lost
  on pod restart/across replicas — fine for a single pod, not for `values-prod.yaml`'s
  `replicaCount: 2` without session affinity. Move it to a `sessions`/`session_messages` table
  in the existing pgvector Postgres (or a separate store if load ever warrants it) so any pod
  can serve any `session_id` and history survives restarts and deploys.
- **Real booking/registration API calls.** Per "Known divergences" above, membership/support
  are advisory-only today. Wiring `MEMBERSHIP_REGISTRATION`'s agent to an actual booking API
  (as a tool, same shape as `search_studio_faq`) would move it from "explains the process" to
  "completes the transaction."
- **Agentic invocation of this API.** Right now every caller is a human via `/chat`. As other
  internal services or agents start calling studio-chatbot programmatically (e.g. an
  orchestrator agent treating this service as one of its own tools), that likely means:
  - a machine-facing auth path (API key/service token) distinct from the human session flow,
  - a stable, documented request/response contract (this README's `/chat` shape becomes a real
    API contract, not just human-readable docs),
  - and probably a dedicated route/flag rather than overloading the same `/chat` + keyword-router
    path both humans and agents hit — an agent calling in doesn't need the keyword short-circuit
    or classifier, it can specify the target agent directly.
- **Session affinity or a shared store, whichever comes first.** Until Postgres-backed sessions
  land, `sessionAffinity: ClientIP` on the Service is a cheap stopgap so `SUMMARIZE`/"resolved"
  reliably hits the pod holding that session's history in prod.
- **Multi-turn tool loop history.** `Assistant.chat()` only sees the current message plus its
  own tool-call loop, not prior turns — once sessions are Postgres-backed, feeding stored
  history into the agent's own messages (not just the summarize agent) would let
  membership/support/general hold a real multi-turn conversation instead of treating each
  message independently.
