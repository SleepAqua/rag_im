# rag_im — immigration policy monitoring

I'm planning to emigrate, and immigration policy is complex, scattered, and
changes without notice. I built this to track it.

Government visa pages are the only trustworthy source, but they are long,
badly structured for machines, and quietly edited. This tool snapshots the
pages you care about, tells you *which section* changed and *what wording*
changed, and answers questions with citations back to the exact page.

It replaced a 48-line LlamaIndex + Chroma script that hardcoded OpenAI
embeddings and only worked with a key and a network connection.

## What it does

1. **Fetch** — download each page in `sources.yaml` with a real User-Agent,
   timeouts and retries, and store the raw HTML under `data/raw/<source_id>/`.
2. **Extract** — strip navigation, headers, footers, asides and scripts; keep
   headings as markdown so the page's structure survives.
3. **Split & hash** — split on headings into sections, hash each section's
   body with whitespace-normalised SHA-256. Reflowing a page is not a change.
4. **Store** — SQLite (`data/policy.sqlite3`): sources, snapshots, sections,
   changes. Every write is idempotent; fetching an unchanged page is a no-op.
5. **Diff** — compare the two most recent snapshots of a source, classify
   sections as added / removed / modified / moved, and write a markdown report
   with before/after snippets and a word-level diff.
6. **Search / ask** — BM25 always works offline; an OpenAI-compatible endpoint
   optionally adds dense retrieval fused with reciprocal-rank fusion, and can
   draft the answer. Citations are always passages that actually exist.

## Architecture

```
                 sources.yaml
                      │
                      ▼
   ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌────────┐
   │ fetch.py │──▶│ extract.py│──▶│sections.py│──▶│store.py│
   └──────────┘   └───────────┘   └───────────┘   └────────┘
        │              HTML→text     split+hash       │  SQLite
        │                                             │
   data/raw/<id>/<ts>.html                    ┌───────┴────────┐
                                              ▼                ▼
                                         ┌────────┐     ┌─────────────┐
                                         │diff.py │     │retrieval.py │
                                         └────────┘     └─────────────┘
                                              │                │ BM25 (+dense, RRF)
                                     reports/*.md              ▼
                                                          ┌─────────┐
                                                          │ ask.py  │
                                                          └─────────┘
                                                               │
                                                          cli.py / __main__.py
```

| Module | Responsibility |
|---|---|
| `config.py` | paths, env settings, `sources.yaml` loading |
| `fetch.py` | HTTP with timeouts/retries/backoff, raw snapshots, orchestration |
| `extract.py` | BeautifulSoup → markdown-ish text, noise removal |
| `sections.py` | heading-anchored sections, whitespace-insensitive hashes |
| `store.py` | SQLite schema, idempotent upserts, query API |
| `diff.py` | change classification, word diff, markdown report |
| `retrieval.py` | BM25, optional embeddings, RRF fusion |
| `ask.py` | grounded extractive or LLM answers with citations |
| `cli.py` | `fetch / diff / search / ask / status` |

## Quickstart

```bash
cd /home/louis/coding/rag_im

# 1. fetch every source in sources.yaml (needs network)
/home/louis/.venvs/main/bin/python -m rag_im fetch

# 2. see what you have
/home/louis/.venvs/main/bin/python -m rag_im status

# 3. compare the two most recent snapshots; writes reports/<id>-<ts>.md
/home/louis/.venvs/main/bin/python -m rag_im diff

# 4. ranked passages with scores
/home/louis/.venvs/main/bin/python -m rag_im search "points requirement" --top-k 5

# 5. a cited answer
/home/louis/.venvs/main/bin/python -m rag_im ask "What is the points requirement for the 189 visa?"
```

Run it from a scheduler (cron/systemd timer) with `fetch`; new snapshots are
diffed automatically and reports land in `reports/`.

### Offline demo

```bash
/home/louis/.venvs/main/bin/python scripts/seed_demo.py --reset
/home/louis/.venvs/main/bin/python -m rag_im search "points requirement"
```

`scripts/seed_demo.py` loads the synthetic fixture pages from `tests/fixtures/`
through the same store/diff code paths, so the whole pipeline can be exercised
with no network.

### Tests

```bash
/home/louis/.venvs/main/bin/python -m unittest discover -s tests -v
```

91 tests, fully offline: extraction noise removal, section splitting, hash
stability under reflow, added/removed/modified/moved classification, BM25
ranking, store idempotency, fetch retries/timeouts, and CLI smoke tests.

## Offline behaviour and what needs a key

Everything except live fetching works with no network and no API key:

- **Fetch** needs network. Timeouts are `(connect=10s, read=30s)` by default and
  retries are bounded, so a dead network fails with a clear message instead of
  hanging.
- **Diff, store, search** are local. BM25 is pure Python — no model download.
- **Ask** returns an extractive answer (the best passages, verbatim, with
  citations) when no endpoint is configured.
- **Dense retrieval and LLM answers** are opt-in:

```bash
export OPENAI_API_KEY=sk-...                        # any OpenAI-compatible key
export RAG_IM_BASE_URL=https://api.openai.com/v1    # optional
export RAG_IM_CHAT_MODEL=gpt-4o-mini                # optional
export RAG_IM_EMBED_MODEL=text-embedding-3-small    # optional
```

With a key, search fuses BM25 and embeddings via RRF; ask sends numbered
passages to chat completions. If the endpoint errors, times out, or cites a
passage number that was never supplied, the code falls back to the extractive
answer — it never prints a citation it cannot back with a stored passage.

## Limitations

- **Sites change their markup.** A CSS selector hint that stops matching falls
  back to whole-document extraction, which is noisier but not fatal. Some
  redesigns will still degrade quality until `sources.yaml` is updated.
- **Some government sites do not serve content to plain HTTP clients.** At the
  time of writing `immi.homeaffairs.gov.au` (AU 189/491) returns a 1.3 MB
  JavaScript shell with an empty `<main>`, and `immigration.govt.nz` returns an
  Incapsula bot wall. The fetcher detects both and reports them as errors
  instead of storing a near-empty snapshot; the raw HTML is kept for diagnosis.
  Monitoring those two sources needs a rendering proxy or a browser-based
  fetcher, which this tool does not ship. The NL and CA pages extract cleanly
  (23 and 4 sections respectively).
- **Snapshots are point-in-time.** A report tells you what changed between two
  fetches, not what the policy was before you started watching. Run `fetch`
  often; gaps are invisible.
- **Hash-based change detection is textual.** A changed number is caught; a
  change in meaning with identical wording is not. Moved-section detection uses
  a hash-prefix heuristic and can pair two genuinely different sections whose
  content happens to be identical.
- **Word-level diffs are noisy for rewrites.** A whole-paragraph rewrite reads
  as a large inline diff; the before/after snippets are usually more useful.
- **Not legal advice.** Always confirm against the official page linked in the
  citation before acting. This tool monitors pages; it does not interpret law.
