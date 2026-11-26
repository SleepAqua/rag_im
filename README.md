# rag_im — immigration policy monitoring

[![tests](https://github.com/SleepAqua/rag_im/actions/workflows/ci.yml/badge.svg)](https://github.com/SleepAqua/rag_im/actions/workflows/ci.yml)

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
   changes, cached embeddings and fetch runs. Every write is idempotent;
   fetching an unchanged page is a no-op.
5. **Diff** — compare the two most recent snapshots of a source, classify
   sections as added / removed / modified / moved, and write a markdown report
   with before/after snippets and a word-level diff.
6. **Search / ask** — BM25 always works offline; an OpenAI-compatible endpoint
   optionally adds dense retrieval fused with reciprocal-rank fusion, and can
   draft the answer. Citations are always passages that actually exist.
7. **Evaluate** — a labelled question set scores retrieval (recall@1/3/5, MRR)
   and gates CI when quality drops.
8. **Run** — `rag-im run` takes a file lock, fetches, diffs, records latency and
   exits non-zero on failures, so it is safe to put behind cron or systemd.

## Architecture

```
                 sources.yaml
                      │
                      ▼
   ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌────────┐
   │ fetch.py │──▶│ extract.py│──▶│sections.py│──▶│store.py│
   └──────────┘   └───────────┘   └───────────┘   └────────┘
        │              HTML→text     split+hash       │  SQLite:
        │                                             │  snapshots, sections,
   data/raw/<id>/<ts>.html                             │  changes, embeddings,
        │                                             │  fetch_runs
        │                    ┌────────────────────────┤
        ▼                    ▼                        ▼
   ┌────────┐          ┌──────────────┐        ┌─────────────┐
   │diff.py │          │retrieval.py  │        │ evaluate.py │
   └────────┘          │BM25 + dense  │        │recall / MRR │
        │              └──────────────┘        └─────────────┘
   reports/*.md               │
                              ▼
                         ┌─────────┐
                         │ ask.py  │
                         └─────────┘
                              │
                    cli.py / runner.py
```

| Module | Responsibility |
|---|---|
| `config.py` | paths, env settings, `sources.yaml` loading |
| `fetch.py` | HTTP with timeouts/retries/backoff, raw snapshots, run records |
| `extract.py` | BeautifulSoup → markdown-ish text, noise removal |
| `sections.py` | heading-anchored sections, whitespace-insensitive hashes |
| `store.py` | SQLite schema, idempotent upserts, embedding cache, metrics |
| `diff.py` | change classification, word diff, markdown report |
| `retrieval.py` | BM25, cached embeddings, RRF fusion |
| `ask.py` | grounded extractive or LLM answers with citations |
| `evaluate.py` | labelled retrieval scoring and CI gate |
| `runner.py` | advisory lock for scheduled runs |
| `cli.py` | `run / fetch / diff / search / ask / eval / status` |

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
rag-im --help
```

Or, from a checkout that already has the dependencies, use the wrapper:

```bash
./run.sh --help
```

## Quickstart

```bash
# 1. fetch every source in sources.yaml (needs network)
./run.sh fetch

# 2. see what you have
./run.sh status

# 3. compare the two most recent snapshots; writes reports/<id>-<ts>.md
./run.sh diff

# 4. ranked passages with scores
./run.sh search "points requirement" --top-k 5

# 5. a cited answer
./run.sh ask "What is the points requirement for the 189 visa?"
```

`fetch` already diffs new snapshots; `diff` is for re-running over what you
already stored. See [`examples/policy-change-report.md`](examples/policy-change-report.md)
for the report format and [`examples/transcript.md`](examples/transcript.md) for
a full offline session.

### Offline demo

```bash
./run.sh seed-demo --reset
export RAG_IM_DB=data/demo.sqlite3   # the demo DB is separate from live state
./run.sh search "points requirement"
./run.sh eval
```

`scripts/seed_demo.py` loads the synthetic fixture pages from `tests/fixtures/`
through the same store/diff code paths, so the whole pipeline can be exercised
with no network.

## Retrieval evaluation

Retrieval quality is measured, not assumed. `eval/questions.yaml` pairs each
question with the exact source and section anchor a correct answer must come
from; a passage that merely mentions the words does not count. The committed
set targets the fixture corpus, so run it against the demo database:

```console
$ RAG_IM_DB=data/demo.sqlite3 ./run.sh eval
Retrieval eval — 17 questions · corpus=11 passages · top_k=5 · mode=bm25
recall@1 0.941   recall@3 1.000   recall@5 1.000   MRR 0.971

No misses in the top 5.
```

For live data, write your own YAML (`id`, `question`, `source`, `anchor`) and
pass `--questions path/to/your.yaml`. If an expected section is absent from the
database, the report says so instead of quietly scoring it as a retrieval miss.

- `--json` emits machine-readable results.
- `--min-recall 0.8` makes CI fail when recall@top-k drops below the threshold.
- `--hybrid` exercises the dense path too; it is off by default because a
  deterministic, free gate is more useful than one that needs an API key.

The fixture corpus is deliberately small (11 passages), so a perfect recall@5
is a regression gate, not evidence of production retrieval quality. Recall@1
and MRR are the numbers that actually move when chunking or ranking regresses.

## Embedding cache

Dense vectors are deterministic for a given model and passage text, so they are
cached in SQLite under `(model, sha256(passage text))`. A query embeds the
question once; it does not re-embed the corpus:

- first query after a corpus change: 11 document embeddings + 1 query
- every later query: 1 query embedding only

`tests/test_embed_cache.py` asserts the batch sizes, so the cost saving is
tested rather than claimed. `--no-cache` bypasses the cache, and
`Store.clear_embeddings()` drops it.

## Scheduling and observability

```bash
./run.sh run
```

`run` fetches under an advisory `flock`, so a slow run cannot overlap the next
timer. Exit codes: `0` all sources succeeded, `1` at least one failed (others
still ran), `3` another run holds the lock, `2` bad `sources.yaml`.

Every attempt is written to the `fetch_runs` table:

```console
$ ./run.sh status --metrics
Fetch runs: 2 total, 2 ok, 0 error, p50=72ms p95=76ms, last=2026-11-25T09:13:38Z

ID             RUNS  OK ERR    P50    P95  LAST RUN             LAST ERROR
au_189            2   2   0    72ms    76ms  2026-11-25T09:13:38Z  -
```

Copy `deploy/rag-im.service` and `deploy/rag-im.timer` for a daily systemd run;
[`deploy/README.md`](deploy/README.md) has the install steps.

## Tests

```bash
./run.sh test
```

123 tests, fully offline: extraction noise removal, section splitting, hash
stability under reflow, added/removed/modified/moved classification, BM25
ranking, RRF, embedding-cache reuse, store idempotency, fetch retries/timeouts,
run locks, eval metrics, and CLI smoke tests. CI runs the suite on Python
3.10–3.13 and runs the eval gate against the demo database.

## Offline behaviour and what needs a key

Everything except live fetching works with no network and no API key:

- **Fetch** needs network. Timeouts are `(connect=10s, read=30s)` by default and
  retries are bounded, so a dead network fails with a clear message instead of
  hanging.
- **Diff, store, search, eval** are local. BM25 is pure Python — no model
  download.
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

## Design decisions

| Decision | Why |
|---|---|
| SQLite instead of Postgres | The working set is a handful of pages; a single file is portable and makes the demo runnable with one command. The schema ports cleanly. |
| Hash section bodies, not headings | A renamed section is a move, not a policy edit. Whitespace is normalised so reflow is invisible. |
| Idempotent writes | Re-running fetch/diff must not invent changes; uniqueness on `(source_id, content_sha256)` and `(snapshot, anchor)` enforces it in the database, not in application code. |
| BM25 before embeddings | Search must work offline and for free; dense retrieval is an optional ranking signal, never a correctness dependency. |
| RRF, not score blending | BM25 and cosine scores are not comparable; rank fusion avoids tuning a weight per corpus. |
| Cache by content hash + model | Same vector, no API call. A chunking or model change naturally misses the cache instead of serving stale vectors. |
| Eval gate in CI | The retrieval path is the easiest part to break silently; a labelled set with recall/MRR makes regressions visible. |

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
  (24 and 4 sections respectively). The gate is structural, not just a
  character count: a shell that clears the length floor with tag-manager
  boilerplate but yields fewer than two sections is rejected too.
- **The eval set is fixture-sized and in-sample.** 17 questions over 11
  passages will not tell you how retrieval behaves on 10,000 documents, and the
  questions were written against the same fixtures used for the demo. It
  catches regressions in chunking and ranking, which is what CI needs; it is
  not a benchmark.
- **Embeddings accumulate.** Vectors are keyed by model and text, so changing
  the chunker or model leaves old rows behind. Call `clear_embeddings()` (or
  delete the table) after a migration; nothing prunes it automatically.
- **Demo data lives in its own database.** `scripts/seed_demo.py` writes
  `data/demo.sqlite3`; mixing fixture snapshots into `data/policy.sqlite3`
  would make the next real fetch look like a wholesale policy rewrite.
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

## License

MIT — see [LICENSE](LICENSE).
