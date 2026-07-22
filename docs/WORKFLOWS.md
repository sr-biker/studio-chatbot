# Workflows

Detailed flowcharts for each of studio-chatbot's code paths. See the root
[README.md](../README.md) for architecture/deployment context — this doc is the
step-by-step "what actually happens" companion to it.

## 1. App startup (`app.main.lifespan`)

Runs once per pod, before any request is served. Note this is a **read-only freshness
check**, not ingestion — see workflow 1b for where embedding actually happens.

```mermaid
flowchart TD
    A[Pod starts] --> B["lifespan() begins"]
    B --> C["check_faq_freshness()"]
    C --> D["_load_markdown_text()\n(Drive export in prod, data/faq.md in local)"]
    D --> E["hash the fetched text"]
    E --> F{hash already\nin pgvector?}
    F -- yes --> G["log.info: in sync"]
    F -- no --> H["log.warning: stale,\nrun the offline ingest job"]
    G --> I["chat_model() built once\n(shared ChatOpenAI instance)"]
    H --> I
    I --> J["Build one Assistant per Route:\nMEMBERSHIP_REGISTRATION, SUPPORT, GENERAL\n(chat_model + FAQ_TOOLS),\nSUMMARIZE (summarize_model, no tools)"]
    J --> K["Router(router_chat_model())"]
    K --> L["SessionStore()"]
    L --> M["yield {assistants, router, sessions}\n-> request.state.*"]
    M --> N[Pod ready to serve traffic]
```

Notes:
- `check_faq_freshness()` never writes to pgvector — it only warns. This keeps
  embedding compute (and its Drive-read dependency) out of the request-serving pod's
  startup critical path; see workflow 1b for the actual ingest.
- `chat_model()`, `router_chat_model()`, `summarize_model()`, `vector_store()`,
  `embedding_model()` are all `@lru_cache`d in `app/ai_config.py` — built once per process,
  reused across every request and every Assistant.

## 1b. Offline FAQ ingestion (`scripts/ingest_faq.py`)

Runs as a Kubernetes `Job` (`helm/templates/ingest-job.yaml`, off by default — see
README's Maintainability Next Step), not from the app pod.

```mermaid
flowchart TD
    A["Job starts\n(python scripts/ingest_faq.py)"] --> B["load_faq_knowledge()"]
    B --> C["_load_markdown_text()"]
    C --> D{FAQ content hash\nalready ingested?}
    D -- yes --> E[No-op, log and skip]
    D -- no --> F["_delete_stale(SOURCE_ID)\n(delete old chunks for this source)"]
    F --> G["MarkdownHeaderTextSplitter\nsplit FAQ by # / ## headers"]
    G --> H["vector_store().add_documents(chunks)"]
    E --> I[Job completes]
    H --> I
```

Notes:
- Idempotency is by **whole-document content hash**, not per-chunk — any FAQ edit
  re-deletes and re-ingests every chunk for that source, not just the changed section.
- Triggered manually today (`helm upgrade ... --set ingestJob.enabled=true`); a
  `CronJob` on a schedule, or a webhook off a Drive change notification, are the
  natural next steps once drift-checking in workflow 1 proves the signal is useful.

## 2. `POST /chat`

The main request path. Every step below runs synchronously per request.

```mermaid
flowchart TD
    A["POST /chat\n{message, session_id?}"] --> B{"is_flagged(message)?\n(OpenAI Moderation API)"}
    B -- flagged --> C["log.warning(...)\nraise HTTPException(400)"]
    B -- clear --> D["session_id = request.session_id\nor uuid4()"]
    D --> E["route = router.route(message)"]

    subgraph ROUTER["Router.route() -- app/router.py"]
        E1{"message empty?"} -- yes --> E2[Route.GENERAL]
        E1 -- no --> E3{"keyword short-circuit\nmatch?"}
        E3 -- membership/join/register/etc --> E4[Route.MEMBERSHIP_REGISTRATION]
        E3 -- summarize/tldr/resolved/etc --> E5[Route.SUMMARIZE]
        E3 -- no match --> E6["LLM classifier\n(temperature=0)"]
        E6 --> E7["parse label ->\nMEMBERSHIP_REGISTRATION / SUPPORT /\nSUMMARIZE / GENERAL\n(unparseable -> GENERAL)"]
    end

    E --> ROUTER
    E7 --> F{route ==\nSUMMARIZE?}
    E2 --> F
    E4 --> F
    E5 --> F

    F -- yes --> G["transcript = sessions.transcript(session_id)"]
    G --> H["assistants[SUMMARIZE].chat(\ntranscript or message)\n(no tools bound)"]
    F -- no --> I["assistants[route].chat(message)"]

    subgraph ASSISTANT["Assistant.chat() -- app/assistant.py"]
        I1["messages = [SystemMessage(prompt),\nHumanMessage(user_message)]"]
        I1 --> I2["model.invoke(messages)"]
        I2 --> I3{"response has\ntool_calls?"}
        I3 -- no --> I4[Return response.content]
        I3 -- yes --> I5["Execute each tool call\n(e.g. search_studio_faq -> pgvector)"]
        I5 --> I6["Append ToolMessage results\nto messages"]
        I6 --> I7{"iteration <\nMAX_TOOL_ITERATIONS (4)?"}
        I7 -- yes --> I2
        I7 -- no --> I8["Return last message's content"]
    end

    I --> ASSISTANT
    H --> J["sessions.append_turn(session_id,\nmessage, reply)"]
    I4 --> J
    I8 --> J
    J --> K["log.info(session_id, route,\nmessage, reply)"]
    K --> L["Return ChatResponse\n{agent, reply, session_id}"]
```

Notes:
- Moderation runs **before** `session_id` resolution and routing — a flagged message never
  reaches the session store, the router, or any model call.
- `SUMMARIZE` is the only route that ignores the literal incoming message for its actual
  model input — it summarizes `sessions.transcript(session_id)` instead, since the trigger
  word itself ("resolved", "tldr") carries no content.
- The tool-calling loop (`Assistant.chat`) is bounded by `MAX_TOOL_ITERATIONS`, not a retry
  mechanism — see README's "Retry transient model-call failures" Next Step for the actual
  retry gap (transient API errors on `.invoke()` itself, not this loop).
- Every turn is logged in one line (`session_id`, `route`, `message`, `reply`) — see
  README's "Observability" section for what's *not* covered (metrics, tracing, alerting).

## 3. `GET /internal/faq/search`

The non-agentic path — no LLM, no router, no session involvement.

```mermaid
flowchart TD
    A["GET /internal/faq/search?q=...&topK=..."] --> B["search_faq_raw(q, topK)"]
    B --> C["k = clamp(topK, 1..MAX_TOP_K)\nor DEFAULT_TOP_K if unset"]
    C --> D["vector_store().similarity_search(q, k)"]
    D --> E["Map hits to\n{section, source, text}"]
    E --> F["Return list[FaqSnippet]"]
```

Notes:
- `search_faq_raw()` is the same function the `search_studio_faq` LangChain tool calls from
  inside the `Assistant.chat()` loop in workflow 2 — this endpoint exists to inspect
  retrieval quality/results directly, independent of what an agent does with them (used by
  `evals/test_ragas_faq.py`'s context-precision checks too).
- No auth today — see README's "Keycloak-based JWT authentication" Next Step.
