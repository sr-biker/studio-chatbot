FROM public.ecr.aws/docker/library/postgres:16

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*

COPY init/01-extensions.sql /docker-entrypoint-initdb.d/
