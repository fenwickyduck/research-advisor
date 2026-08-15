# Research Advisor

A personal reading advisor for research papers. You tell it what you've read, it recommends
what to read next, you tell it what you thought, and the recommendations improve.

Corpus: **arXiv** (configurable categories, `cs.CR` by default) and the
**Cryptology ePrint Archive**. Recommendations are hybrid — local embeddings retrieve
candidates, and each one is attributed to the paper of yours that pulled it in.
It runs entirely on your machine: no API key, no account, no per-run cost.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Use

```sh
advisor add 2401.12345 2024/123 10.1007/3-540-48910-X_16   # papers you've read
advisor harvest                                            # pull the corpus to recommend from
advisor embed                                              # encode it (resumable)
advisor recommend                                          # what to read next
advisor profile                                            # what the advisor thinks you work on
advisor serve                                              # http://127.0.0.1:8000
advisor status                                             # what's in the database

advisor update                                             # all of the above, for a cron job
advisor schedule                                           # how to run that nightly
advisor reset                                              # forget what it learned about you
```

`add` accepts arXiv IDs, ePrint IDs, DOIs, and any URL containing one — one per line on
stdin, or as arguments. The same works from the browser at `/add`.

`recommend` records the batch it shows you, exactly as the browser feed does, so nothing
is recommended twice. Use `--preview` to look without recording — otherwise the terminal
would keep repeating one list while the feed moved on.

`harvest` is incremental: it records a cursor per source, so the first run backfills
(arXiv `cs.CR` alone is ~51,000 papers, plus ~27,000 from ePrint) and every run after
that fetches only what changed. It is safe to interrupt — progress is saved per page, and
rerunning resumes.

### Running unattended

`advisor update` is the whole nightly job — harvest, embed, then build a fresh feed.
The three belong together: harvesting without embedding leaves the new papers invisible
to retrieval, and embedding without a run leaves them out of the feed until you next
press the button.

It is written to be safe on a timer rather than merely automatic:

- **The embed pass is bounded** (`--max-minutes`, default 30). The first run after a
  backfill would otherwise hold the machine for hours; instead it stops at the next
  batch boundary and the remainder is picked up the following night.
- **Passes cannot overlap.** `advisor embed` and a scheduled `advisor update` take an
  advisory lock, because both rewrite the vector matrix wholesale — the loser would
  silently discard the winner's work. A scheduled run that finds the lock held says so
  and moves on rather than queueing behind an hours-long manual pass.

`advisor schedule` prints a ready-made systemd timer and a crontab line, pointing at
this installation's absolute path (both cron and systemd run with a minimal `PATH` that
will not find a virtualenv). It installs nothing — paste what you want:

```sh
advisor schedule            # systemd units + crontab line, defaults to 06:00
advisor schedule --at 23:30
```

### Starting over

`advisor reset` clears your library, ratings, profile and recommendation history, and
keeps the harvested corpus and its vectors — the half that costs hours to rebuild.
`advisor reset --all` empties everything. Both confirm before deleting; `--yes` skips
the prompt for scripts.

### Feedback

A thumbs-down opens optional reason chips and a note box. The reason is not
decoration — it changes how far the rejection moves your interests:

| reason | effect on retrieval |
|---|---|
| I already know this | none — retrieval was right, the paper is just redundant |
| Wrong subfield | full push away |
| Too theoretical / applied / not rigorous | half push |
| Want newer work | leaves interests alone, leans the recency prior (capped at 3x) |

An unlabelled thumbs-down gets the full default push, so explaining yourself stays
optional. The free-text note is kept for the interest profile, where it
carries the signal a vector cannot — "I want attacks, not surveys".

### Ranking and the interest profile

Ranking is cosine similarity, diversified by MMR, with each pick attributed to the
paper of yours it is nearest to. There is no model call anywhere in it.

The profile is the piece that makes this an advisor rather than a similarity search.
Embeddings capture "more like this"; only words carry "I already know RLWE hardness,
stop showing me introductions". You write it — at `/profile`, in the browser — and
every version is kept, so you can see how the picture of you changed.

#### The profile steers retrieval

Two of its sections are instructions rather than description, and retrieval reads them
directly:

```markdown
## More of
doubly-efficient private information retrieval
zero-knowledge proof systems and succinct arguments

## Less of
post-quantum migration policy surveys
```

Each line is encoded by the same local model that encodes the corpus — SPECTER takes
arbitrary text, not only papers — so a line under **More of** becomes a query vector and
a line under **Less of** becomes something to steer away from, at the same full strength
as a "wrong subfield" rejection. This is the one part of the advisor you drive with
words instead of clicks: type it at `/profile` and it applies to your next batch.

Write them like paper titles, not sentences about yourself. That is what the encoder was
trained on, and it shows — the top corpus matches for `lattice-based digital signatures`
are CRYSTALS-Dilithium, *Sharper Ring-LWE Signatures* and *Asymptotically Efficient
Lattice-Based Digital Signatures*; for `side-channel attacks on AES implementations`,
*Algebraic Side-Channel Collision Attacks on AES* and *Masking of AES*.

A stated interest is one interest among your others, not an override: it takes its share
of the results alongside what you have read, weighted by `profile_weight`. A profile is
also enough on its own — with an empty library, saying what you want is a complete
starting point.

The remaining sections stay prose, for you to read. A profile that is all prose simply
steers nothing; `/profile` tells you which case you are in.

#### If you would rather not write it yourself

`advisor profile --brief` (and a panel at `/profile`) prints your reading, your ratings
and your notes, together with instructions for turning them into a profile. Copy it into
whatever assistant you already have — Claude on the web, say — and paste the answer back
into the editor. You get a written profile without this program holding a credential or
spending anything, and because it is text you carry by hand, you can read exactly what
you are sending before you send it.

There is deliberately no button that does this for you. **The advisor holds no API key,
reads no credential from the environment, and makes no network call except to arXiv,
ePrint and Crossref when harvesting.** `tests/test_no_api.py` asserts all three, so
re-adding one has to be a decision somebody makes on purpose rather than a dependency
that creeps back.

#### Why every recommendation still comes with a reason

```
3. [0.943] (2026) GPU Acceleration of Learning With Errors KEMs Using OpenACC
      Closest to “Portable Acceleration of Lattice KEMs for Post-Quantum
      Cryptography”, which you read (cosine 0.95).
```

Naming the paper that pulled a recommendation in is the honest answer to "why am I
being shown this" — it is literally what drove the match, it costs one matrix product,
and it is often easier to check than a written justification, because you recognise the
paper it names. Papers you have explicitly disliked are never offered as the reason for
a new one.

### How it stays current

Both sources are harvested over OAI-PMH, whose `from` parameter filters on **modification**
date rather than submission date. That matters more than it sounds:

- **Revisions are picked up.** When an author posts a v2 with a rewritten abstract, the
  record comes back and the stored abstract is replaced. Since the abstract is what gets
  embedded, harvesting on submission date instead would leave recommendations computed
  from text the author had already replaced.
- **Late cross-listings appear.** A paper moved into `cs.CR` months after submission shows
  up on the next run.
- **Withdrawals are recorded.** Deleted records set `withdrawn_at`, so the paper stops
  being recommended — but the row survives, because a paper already in your library should
  not vanish from your history.

Merging follows two rules. A record arriving again **from the same source** is
authoritative and replaces the stored content. A record matched **across sources** (same
title and an overlapping author) fills in gaps and adds its identifier, but never
overwrites a fuller record with a terser one — and category lists always union, so a paper
on both arXiv and ePrint keeps both taxonomies.

Data lives in `~/.local/share/advisor/`; settings, if you want any, go in
`~/.config/advisor/config.toml` (see `advisor/config.py` for the keys and defaults).

## Status

| Phase | | |
|---|---|---|
| 1 | Skeleton, library, ID resolution | **done** |
| 2 | Corpus harvest from arXiv + ePrint | **done** |
| 3 | Embeddings and vector retrieval | **done** |
| 4 | Feedback loop | **done** |
| 5 | The interest profile and profile-steered retrieval | **done** |
| 6 | Scheduling and polish | **done** |

Phase 3 adds dependencies deliberately kept out of the base install:

```sh
.venv/bin/pip install -e '.[embed]'   # sentence-transformers, onnxruntime, numpy, scikit-learn
```

### Embedding: you do not wait for it

`advisor embed` encodes the corpus locally with SPECTER. The full 76k papers take roughly
3 hours on CPU, but the tool is useful within a couple of minutes and never blocks on it:

- Work is ordered **library first, then a spread across years** — the newest paper of every
  year, then the second newest, and so on. Partial results are therefore representative
  rather than skewed to whatever is currently fashionable.
- It is resumable. Stop it, rerun it, it picks up where it left off.
- Recommendations run against whatever is embedded so far and simply get deeper.
- It defaults to an int8 ONNX graph, measured 4.5x faster than torch fp32 (2.4 -> 10.9
  papers/s) for vectors agreeing at 0.988 mean cosine and ~85% top-10 neighbour overlap.
  Set `embedding_backend = "torch"` for exact fp32 vectors at 4x the wall clock.

After the first pass, a day's new papers take about ten seconds.

## Tests

```sh
.venv/bin/python -m pytest
```

Parser tests run against saved API responses in `tests/fixtures/`, so the suite needs no
network.

## Attribution

Paper metadata comes from the [arXiv API](https://arxiv.org), the
[Cryptology ePrint Archive](https://eprint.iacr.org) (© IACR and the respective authors,
harvested via OAI-PMH under its stated terms), and [Crossref](https://www.crossref.org).
