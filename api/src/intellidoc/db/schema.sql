CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,          -- first 16 hex chars of the file's SHA-256
    filename    text NOT NULL,
    title       text NOT NULL,
    page_count  integer NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pages (
    doc_id          text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page            integer NOT NULL,
    method          text NOT NULL CHECK (method IN ('native', 'ocr')),
    native_chars    integer NOT NULL,
    ocr_confidence  real,
    PRIMARY KEY (doc_id, page)
);

-- One row per chunk per chunking strategy, so strategies can be compared on the same parse.
CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    doc_id       text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunker      text NOT NULL,
    ordinal      integer NOT NULL,
    pages        integer[] NOT NULL,
    kind         text NOT NULL,
    section      text NOT NULL,
    text         text NOT NULL,
    search_text  text NOT NULL,
    metadata     jsonb NOT NULL,
    embedding    vector(384) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    UNIQUE (doc_id, chunker, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_chunker_idx ON chunks (chunker);
-- No ANN index on embedding: see docs/adr/0002-exact-vector-search.md.
