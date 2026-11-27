# Transcript (offline demo data)

Commands below use the seeded demo database, so they work with no network and
no API key.

```console
$ ./run.sh seed-demo --reset
report: reports/au_189-20261031T090000Z.md
...
Summary: 1 added, 1 removed, 2 modified, 1 moved
$ export RAG_IM_DB=data/demo.sqlite3
```

## Search

```console
$ ./run.sh search "points requirement" --top-k 3
1. score  3.535  AU · Skilled Independent visa (subclass 189) · Points requirement
   You must score at least 67 points on the points test to be eligible for an invitation. Points are awarded for age, English language ability, skilled employment experience and qualifications. Invitations are issued to the highest scoring can…
   https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189
2. score  1.300  AU · Skilled Independent visa (subclass 189) · Skilled Independent visa (subclass 189)
   This visa lets invited skilled workers live and work permanently anywhere in Australia. It is points-tested and does not require an employer sponsor.…
   https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189
3. score  1.136  AU · Skilled Independent visa (subclass 189) · Changes from 1 July 2026
   From 1 July 2026 the minimum age for invited applicants drops from 45 to 40 and the points pass mark rises to 67. Applicants invited before that date are assessed under the previous settings.…
   https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189
```

## Ask (extractive, no key)

```console
$ ./run.sh ask "What is the points requirement for the 189 visa?" --extractive --top-k 2
Q: What is the points requirement for the 189 visa?

Extractive answer (LLM not used) — the most relevant passages, verbatim:

[1] AU · Skilled Independent visa (subclass 189) — "Points requirement"
You must score at least 67 points on the points test to be eligible for an invitation. Points are awarded for age, English language ability, skilled employment experience and qualifications. Invitations are issued to the highest scoring candidates in each occupation, so in practice the competitive score is often higher than the pass mark.
    source: https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189

[2] AU · Skilled Independent visa (subclass 189) — "Skilled Independent visa (subclass 189)"
This visa lets invited skilled workers live and work permanently anywhere in Australia. It is points-tested and does not require an employer sponsor.
    source: https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189

Citations:
  [1] AU · Skilled Independent visa (subclass 189) — “Points requirement” — https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189
  [2] AU · Skilled Independent visa (subclass 189) — “Skilled Independent visa (subclass 189)” — https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189

(mode: extractive)
```

## Retrieval eval

```console
$ ./run.sh eval
Retrieval eval — 17 questions · corpus=11 passages · top_k=5 · mode=bm25
recall@1 0.941   recall@3 1.000   recall@5 1.000   MRR 0.971

No misses in the top 5.
```

## Status

```console
$ ./run.sh status
DB: /home/louis/coding/rag_im/data/demo.sqlite3
sources=2 snapshots=3 sections=18 changes=5 embeddings=0
model: no endpoint configured · BM25 only

ID            CC    SNAPS  LAST FETCH           LAST CHANGE
au_189        AU        2  2026-10-31T09:00:00Z moved: English language requirements
nl_hsm        NL        1  2026-09-15T12:00:00Z -
```
