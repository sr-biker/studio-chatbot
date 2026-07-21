# studio-chatbot

FastAPI + LangChain chatbot for a fitness studio. FAQ knowledge (`s3://senthil-studio-faq/faq.md`)
is chunked and embedded into pgvector for RAG; an incoming message is routed to one of three
named agents — **membership_registration**, **support**, **general** — each grounded in the
FAQ via a shared retrieval tool. Structurally a port of the `premed-python` pattern
(`config`/`ai_config`/`router`/`assistant`/`main` split, migrations, compose, Dockerfile) with
the domain swapped from pre-med admissions to a gym.

## Architecture

```
POST /chat --> Router (keyword short-circuit, then LLM classifier)
                 |
                 +--> MEMBERSHIP_REGISTRATION agent --> search_studio_faq tool --> pgvector
                 +--> SUPPORT agent                 --> search_studio_faq tool --> pgvector
                 +--> GENERAL agent                  --> search_studio_faq tool --> pgvector
```

- `app/faq_loader.py` — pulls `faq.md` (S3 in prod, checked-in `data/faq.md` copy in local),
  splits it by markdown header (one chunk per FAQ section), and ingests into pgvector. Idempotent
  on a content hash — re-running with unchanged FAQ text is a no-op; changed text deletes and
  re-ingests.
- `app/router.py` — cheap keyword short-circuit (join/register/membership/etc → membership
  registration) then a temperature-0 LLM classifier for anything ambiguous.
- `app/agents/` — one system prompt per named agent, all sharing the same FAQ retrieval tool
  and tool-calling loop (`app/assistant.py`).

## Embedding models: local vs. prod

Both profiles use a same-family embedding model, but **not the same vectors or table** — a
pgvector column is fixed-dimension, so local (`sentence-transformers/all-MiniLM-L6-v2`,
384-dim, in-process, no API key needed) and prod (`text-embedding-3-small`, 1536-dim, OpenAI)
each get their own store, re-embedded from the same FAQ source. There is no cross-env vector
sharing.

## Providers

| Component            | local                          | prod                                          |
|-----------------------|---------------------------------|------------------------------------------------|
| Chat / agents / router | OpenAI (`gpt-4o-mini`)         | AWS Bedrock (`anthropic.claude-3-5-sonnet-*`)  |
| Embeddings              | HuggingFace (MiniLM, local)    | OpenAI (`text-embedding-3-small`)              |
| Vector store            | pgvector (docker compose)      | pgvector (helm, on k8s)                        |

Prod chat needs no `OPENAI_API_KEY` for the model itself, but embeddings still call OpenAI —
both `OPENAI_API_KEY` and AWS credentials (for Bedrock + S3 + optionally Secrets Manager) are
required in prod.

## Local development

```bash
docker compose up -d --build      # build the app image and start app + postgres(+pgvector)
docker compose logs -f app
docker compose down                # add -v to also drop the pgdata volume
```

Or directly:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
export $(cat .env | xargs)
uvicorn app.main:app --reload --port 8080
```

```bash
curl -s -X POST http://localhost:8089/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How do I sign up for the yoga class?"}'

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

`.github/workflows/ci.yml` runs `tests` on every push/PR, and `evals` (against a
`pgvector/pgvector:pg16` service container) on pushes to `main`.

## Kubernetes / Helm

Two charts, deployed separately (`helm/pgvector` first, `helm/studio-chatbot` second — the app
expects `DB_HOST` to resolve to the pgvector chart's release name):

```bash
helm install pgvector helm/pgvector -f helm/pgvector/values.yaml
helm install studio-chatbot helm/studio-chatbot -f helm/studio-chatbot/values.yaml

# prod
helm install pgvector helm/pgvector -f helm/pgvector/values-prod.yaml
helm install studio-chatbot helm/studio-chatbot -f helm/studio-chatbot/values-prod.yaml \
  --set secrets.dbPassword=... \
  --set config.dbSecretName=/rds/... \
  --set config.awsRegion=us-east-1
```

Notes:
- `pgvector` is a single-replica `StatefulSet` with a `PersistentVolumeClaim` — not HA, matches
  the scale of a single FAQ document. Not the `~/projects/infra` RDS module; this app owns its
  own DB lifecycle in-cluster (see the pgvector-location tradeoff discussed with the user before
  building this).
- In prod, the app pod needs IAM permissions for Bedrock (`bedrock:InvokeModel`), S3
  (`s3:GetObject` on `senthil-studio-faq/faq.md`), and optionally Secrets Manager — attach via
  IRSA (`serviceAccountName` + annotated `ServiceAccount`, cluster/account-specific, not
  templated here) rather than static keys.
- `values-prod.yaml` in both charts has `<ECR_REPO_URL>` placeholders and no real secrets —
  fill in at deploy time from wherever this org keeps prod credentials, never check them in.

## Known divergences / simplifications

- "Membership registration" and "support" are chat *advisory* agents grounded in the FAQ, not a
  transactional booking system — matches the FAQ's own guidance ("register via the member
  portal, mobile app, or front desk"); this bot explains policy and process, it doesn't call a
  booking API (there isn't one in scope here).
- Evals are offline/CI-gated only, not runtime guardrails — see "Evals" above.
