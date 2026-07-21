-- Initial schema for the studio (gym) chatbot.

CREATE EXTENSION IF NOT EXISTS vector;

-- `langchain-postgres`'s PGVector integration manages its own collection/embedding tables
-- (langchain_pg_collection / langchain_pg_embedding) and creates them itself on first use —
-- nothing to migrate here for the FAQ vector store. {vector_dimension} (384 local /
-- 1536 prod) is enforced implicitly by whichever embedding model is configured.
